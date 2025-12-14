import re
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

from prompts import infer_prompt

try:
    from accelerate import infer_auto_device_map, init_empty_weights
    from accelerate.utils import get_balanced_memory
except Exception as e:
    raise ImportError(
        "This script requires accelerate. Please install it via:\n"
        "  pip install accelerate\n"
    ) from e

STOP_TAGS_FOR_STOPPING = ["<search>", "</search>", "</backtrack>", "</summary>", "</answer>"]
CLOSE_TAGS = ["</search>", "</backtrack>", "</summary>", "</answer>"]

MODEL_TAGS = ["think", "answer", "summary", "backtrack", "search"]


def Search(_: str) -> str:
    return "Cannot Search Now"


def configure_tf32_no_deprecated_warning() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


class ActionTagStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer: Any,
        stop_tags: List[str],
        max_prefix_ws: int = 2,
        max_suffix_nl: int = 2,
        include_crlf: bool = True,
        add_special_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_tags = stop_tags
        self.triggered_tag: Optional[str] = None

        ws_prefixes = [""]
        for k in range(1, max_prefix_ws + 1):
            ws_prefixes.append(" " * k)
            ws_prefixes.append("\n" * k)
        ws_prefixes = list(dict.fromkeys(ws_prefixes))

        suffixes = [""]
        for k in range(1, max_suffix_nl + 1):
            suffixes.append("\n" * k)
        if include_crlf:
            suffixes += ["\r\n", "\r\n\r\n"]
        suffixes = list(dict.fromkeys(suffixes))

        buckets: Dict[int, List[Tuple[str, torch.Tensor]]] = {}
        for tag in stop_tags:
            for pre in ws_prefixes:
                for suf in suffixes:
                    text = f"{pre}{tag}{suf}"
                    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
                    if not ids:
                        continue
                    L = len(ids)
                    buckets.setdefault(L, []).append((tag, torch.tensor(ids, dtype=torch.long)))

        self.lengths = sorted(buckets.keys())
        self.buckets = buckets

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        self.triggered_tag = None
        _, seq_len = input_ids.shape
        device = input_ids.device

        for L in self.lengths:
            if seq_len < L:
                continue
            tail = input_ids[:, -L:]
            for tag, ids_cpu in self.buckets[L]:
                ids = ids_cpu.to(device)
                if (tail == ids).all(dim=1).any():
                    self.triggered_tag = tag
                    return True
        return False


@dataclass
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = False


@dataclass
class SearchConfig:
    num_paths: int = 3
    max_new_tokens: int = 96
    temperature: float = 1.0
    top_p: float = 0.5
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3


class ProtocolRunner:
    def __init__(
        self,
        model_dir: str,
        dtype: str = "bf16",
        attn_impl: str = "sdpa",
        reserve_gib: float = 1.5,
        balanced_low_0: bool = False,
        search_cfg: Optional[SearchConfig] = None,
    ):
        self.model_dir = model_dir
        self.search_cfg = search_cfg or SearchConfig()

        configure_tf32_no_deprecated_warning()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            use_fast=True,
        )

        if dtype == "auto":
            torch_dtype = None
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "fp32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

        n_gpu = torch.cuda.device_count()
        if n_gpu == 0:
            raise RuntimeError("No CUDA device found, cannot run fully on GPU.")

        reserve = int(reserve_gib * 1024**3)
        max_memory: Dict[int, int] = {}
        for i in range(n_gpu):
            total = torch.cuda.get_device_properties(i).total_memory
            avail = max(total - reserve, int(0.5 * 1024**3))
            max_memory[i] = avail
            
        with init_empty_weights():
            empty_model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
            )

        balanced_mem = get_balanced_memory(
            empty_model,
            max_memory=max_memory,
            dtype=torch_dtype,
            low_zero=balanced_low_0,
        )

        inferred_map = infer_auto_device_map(
            empty_model,
            max_memory=balanced_mem,
            no_split_module_classes=["Qwen2DecoderLayer", "Qwen2.5DecoderLayer", "DecoderLayer"],
        )

        bad = [(k, v) for k, v in inferred_map.items() if v in ("cpu", "disk")]
        if bad:
            raise RuntimeError(
                "FULL-GPU requirement not satisfied: some modules would be placed on cpu/disk.\n"
                f"Offloaded modules: {bad}\n"
                "Fix options:\n"
                "  1) Free GPU memory / close other GPU jobs\n"
                "  2) Reduce reserve_gib (careful about OOM)\n"
                "  3) Use quantized weights (int8/int4) but still GPU-only\n"
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
            device_map=inferred_map,
        )
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dev = next(self.model.parameters()).device
        print(f"[Init] first param device: {dev}")
        if hasattr(self.model, "hf_device_map"):
            print(f"[Init] hf_device_map: {self.model.hf_device_map}")

    def _wrap_prompt(self, query: str) -> str:
        return infer_prompt + f"\n<user_query>{query}</user_query>\n"

    @staticmethod
    def _parse_actions_from_pending(pending: str) -> Tuple[List[Tuple[str, str]], str]:
        """
        Parse as many COMPLETE actions as possible from pending text.
        Returns:
          actions: List[(tag, full_text)]
          rest: remaining tail (incomplete) to carry over
        """
        actions: List[Tuple[str, str]] = []
        i = 0
        n = len(pending)

        open_pat = re.compile(r"<(think|answer|summary|backtrack|search)>", re.DOTALL)

        while i < n:
            m = open_pat.search(pending, i)
            if not m:
                return actions, pending[i:]

            start = m.start()
            tag = m.group(1)
            close = f"</{tag}>"
            close_idx = pending.find(close, m.end())
            if close_idx == -1:
                return actions, pending[start:]

            end = close_idx + len(close)
            actions.append((tag, pending[start:end]))
            i = end

        return actions, ""

    @staticmethod
    def _extract_tag_payload(action_text: str, tag: str) -> str:
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        m = re.search(pattern, action_text, flags=re.DOTALL)
        return m.group(1) if m else ""

    @staticmethod
    def _strip_path_markers(text: str) -> str:
        text = re.sub(r"\[/?path[^\]]*\]", "", text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _delete_prev_model_action(traj: List[Dict[str, str]]) -> None:
        """
        Delete the MODEL action immediately BEFORE the last MODEL action.
        Used for <summary> / <backtrack>.
        """
        for i in range(len(traj) - 2, -1, -1):
            if traj[i]["kind"] == "MODEL":
                traj.pop(i)
                break
            else:
                traj.pop(i)

    def _run_parallel_search_paths(
        self,
        prefix_text: str,
    ) -> Tuple[str, str]:
        cfg = self.search_cfg
        device = self.model.device

        enc = self.tokenizer([prefix_text], return_tensors="pt", padding=True, padding_side="left")
        prefix_ids = enc.input_ids.to(device, non_blocking=True)
        prefix_mask = enc.attention_mask.to(device, non_blocking=True)
        prefix_len = prefix_ids.size(1)


        stopper = ActionTagStoppingCriteria(
                tokenizer=self.tokenizer,
                stop_tags=STOP_TAGS_FOR_STOPPING,
                max_prefix_ws=2,
                max_suffix_nl=2,
                include_crlf=True,
            )

        with torch.inference_mode():
            out = self.model.generate(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                num_return_sequences=cfg.num_paths,
                repetition_penalty=cfg.repetition_penalty,
                no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([stopper]),
                use_cache=True,
            )

        search_lines: List[str] = []
        obs_lines: List[str] = []

        for idx in range(out.size(0)):
            full_seq = out[idx]
            gen_part = full_seq[prefix_len:]
            gen_text = self.tokenizer.decode(gen_part, skip_special_tokens=False)

            end_idx = gen_text.find("</search>")
            if end_idx != -1:
                gen_text = gen_text[:end_idx]

            gen_text = self._strip_path_markers(gen_text)

            if "<search>" in gen_text:
                gen_text = gen_text.split("<search>")[-1].strip()

            path_name = f"path{idx + 1}"
            search_lines.append(f"[{path_name}] {gen_text} [/{path_name}]")

            try:
                obs = Search(gen_text)
            except Exception as exc:
                obs = f"Error: {exc}"
            
            obs = self._strip_path_markers(str(obs))
            obs_lines.append(f"[{path_name}] {obs} [/{path_name}]")

        final_search_block = "<search>\n" + "\n".join(search_lines) + "\n</search>"
        final_obs_block = "<observation>\n" + "\n".join(obs_lines) + "\n</observation>\n"
        return final_search_block, final_obs_block

    def Generate(self, query: str, N: int, gen_cfg: Optional[GenConfig] = None) -> str:
        if gen_cfg is None:
            gen_cfg = GenConfig()

        base_context = self._wrap_prompt(query)

        # traj: list of {"kind": "MODEL"/"OBS", "tag": ..., "text": ...}
        traj: List[Dict[str, str]] = []
        pending_text = ""
        interruptions = 0
        
        stopper = ActionTagStoppingCriteria(
                tokenizer=self.tokenizer,
                stop_tags=STOP_TAGS_FOR_STOPPING,
                max_prefix_ws=2,
                max_suffix_nl=2,
                include_crlf=True,
            )
        stopping = StoppingCriteriaList([stopper])

        def traj_texts() -> List[str]:
            return [x["text"] for x in traj]

        while True:
            if interruptions >= N:
                return "".join(traj_texts()) + pending_text
            
            interruptions += 1
            prompt_text = base_context + "".join(traj_texts()) + pending_text
            print(f"ITER-{interruptions}:\n {prompt_text}")

            inputs = self.tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device, non_blocking=True)
            attention_mask = inputs["attention_mask"].to(self.model.device, non_blocking=True)
            input_len = input_ids.shape[1]

            with torch.inference_mode():
                gen_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_cfg.max_new_tokens,
                    do_sample=gen_cfg.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    stopping_criteria=stopping,
                    use_cache=True,
                )
                if gen_cfg.do_sample:
                    gen_kwargs.update(dict(temperature=gen_cfg.temperature, top_p=gen_cfg.top_p))

                out_ids = self.model.generate(**gen_kwargs)

            decoded = self.tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=False)
            
            pending_text += decoded
            
            new_actions, pending_text = self._parse_actions_from_pending(pending_text)
            
            for tag, full_text in new_actions:
                traj.append({"kind": "MODEL", "tag": tag, "text": full_text})

                if tag == "answer":
                    return "".join(traj_texts())

                if tag in ("summary", "backtrack"):
                    self._delete_prev_model_action(traj)
                    break

            open_pos = pending_text.find("<search>")
            if open_pos != -1:
                prefix_for_search = base_context + "".join(traj_texts()) + pending_text[:open_pos] + "<search>"
                final_search_block, final_obs_block = self._run_parallel_search_paths(prefix_for_search)

                traj.append({"kind": "MODEL", "tag": "search", "text": final_search_block})
                traj.append({"kind": "OBS", "tag": "observation", "text": final_obs_block})

                pending_text = ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="/root/zzx/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--N", type=int, default=8, help="最大中断次数")

    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--no_sample", action="store_true")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--reserve_gib", type=float, default=1.5, help="per-GPU VRAM reserve in GiB")
    ap.add_argument("--balanced_low_0", action="store_true", help="accelerate low_zero mode")

    # parallel search knobs
    ap.add_argument("--search_paths", type=int, default=3)
    ap.add_argument("--search_max_new_tokens", type=int, default=96)
    ap.add_argument("--search_temperature", type=float, default=1.0)
    ap.add_argument("--search_top_p", type=float, default=0.5)
    ap.add_argument("--search_repetition_penalty", type=float, default=1.1)
    ap.add_argument("--search_no_repeat_ngram", type=int, default=3)

    args = ap.parse_args()

    search_cfg = SearchConfig(
        num_paths=args.search_paths,
        max_new_tokens=args.search_max_new_tokens,
        temperature=args.search_temperature,
        top_p=args.search_top_p,
        repetition_penalty=args.search_repetition_penalty,
        no_repeat_ngram_size=args.search_no_repeat_ngram,
    )

    runner = ProtocolRunner(
        model_dir=args.model_dir,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        reserve_gib=args.reserve_gib,
        balanced_low_0=args.balanced_low_0,
        search_cfg=search_cfg,
    )

    gen_cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=(not args.no_sample),
    )

    while True:
        try:
            q = input("Query> ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        out = runner.Generate(q, N=args.N, gen_cfg=gen_cfg)
        print(out, flush=True)


if __name__ == "__main__":
    main()
