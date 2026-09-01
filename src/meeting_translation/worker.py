from __future__ import annotations

import heapq
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(order=True)
class InferenceTask:
    priority: int
    sequence: int
    kind: Literal["asr", "translate"] = field(compare=False)
    segment_id: str = field(compare=False)
    revision: int = field(compare=False)
    final: bool = field(compare=False, default=False)
    model_name: str = field(compare=False, default="")
    audio: np.ndarray | None = field(compare=False, default=None)
    text: str = field(compare=False, default="")
    audio_start_ms: int = field(compare=False, default=0)
    audio_end_ms: int = field(compare=False, default=0)
    submitted_monotonic: float = field(compare=False, default_factory=time.monotonic)


@dataclass(frozen=True)
class InferenceResult:
    kind: Literal["asr", "translate", "error", "ready", "superseded"]
    segment_id: str
    revision: int
    final: bool
    text: str
    model_name: str
    audio_start_ms: int
    audio_end_ms: int
    elapsed_seconds: float
    queue_seconds: float
    gpu_memory_mb: int
    error: str = ""


@dataclass(frozen=True)
class WorkerSettings:
    primary_asr: str
    fallback_asr: str
    translator: str
    prompt_terms: str
    sample_rate: int = 16_000


class GPUWorkerClient:
    def __init__(self, settings: WorkerSettings) -> None:
        context = mp.get_context("spawn")
        self.input_queue: mp.Queue = context.Queue()
        self.output_queue: mp.Queue = context.Queue()
        self.process = context.Process(
            target=_worker_main,
            args=(settings, self.input_queue, self.output_queue),
            name="meeting-translation-gpu",
            daemon=True,
        )
        self.sequence = 0

    def start(self) -> None:
        self.process.start()

    def submit_asr(
        self,
        segment_id: str,
        revision: int,
        final: bool,
        model_name: str,
        audio: np.ndarray,
        start_ms: int,
        end_ms: int,
    ) -> None:
        self.sequence += 1
        self.input_queue.put(InferenceTask(
            priority=0,
            sequence=self.sequence,
            kind="asr",
            segment_id=segment_id,
            revision=revision,
            final=final,
            model_name=model_name,
            audio=np.asarray(audio, dtype=np.float32),
            audio_start_ms=start_ms,
            audio_end_ms=end_ms,
        ))

    def submit_translation(self, segment_id: str, revision: int, text: str, start_ms: int, end_ms: int) -> None:
        self.sequence += 1
        self.input_queue.put(InferenceTask(
            priority=1,
            sequence=self.sequence,
            kind="translate",
            segment_id=segment_id,
            revision=revision,
            final=True,
            text=text,
            audio_start_ms=start_ms,
            audio_end_ms=end_ms,
        ))

    def get_result(self, timeout: float = 0.2) -> InferenceResult | None:
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def alive(self) -> bool:
        return self.process.is_alive()

    def close(self) -> None:
        if self.process.is_alive():
            self.input_queue.put(None)
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.input_queue.close()
        self.input_queue.join_thread()
        self.output_queue.close()
        self.output_queue.join_thread()


def _remove_unsupported_generation_inputs(inputs):
    inputs.pop("token_type_ids", None)
    return inputs

def _contains_cjk(text: str) -> bool:
    return any("\u3400" <= character <= "\u9fff" for character in text)


def _startup_asr_models(settings: WorkerSettings) -> tuple[str, ...]:
    return (settings.primary_asr,)


def _coalesce_pending(tasks: list[InferenceTask]) -> tuple[list[InferenceTask], list[InferenceTask]]:
    latest_asr: dict[str, InferenceTask] = {}
    retained: list[InferenceTask] = []
    dropped: list[InferenceTask] = []
    for task in tasks:
        if task.kind != "asr":
            retained.append(task)
            continue
        previous = latest_asr.get(task.segment_id)
        if previous is None:
            latest_asr[task.segment_id] = task
        elif (task.revision, task.sequence) > (previous.revision, previous.sequence):
            dropped.append(previous)
            latest_asr[task.segment_id] = task
        else:
            dropped.append(task)
    retained.extend(latest_asr.values())
    heapq.heapify(retained)
    return retained, dropped


def _worker_main(settings: WorkerSettings, input_queue: mp.Queue, output_queue: mp.Queue) -> None:
    import torch
    from qwen_asr import Qwen3ASRModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.backends.cuda.matmul.allow_tf32 = True
    asr_models: dict[str, object] = {}
    translator = None
    tokenizer = None
    pending: list[InferenceTask] = []

    def gpu_memory() -> int:
        return int(torch.cuda.memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0

    def load_asr(model_name: str):
        if model_name not in asr_models:
            asr_models[model_name] = Qwen3ASRModel.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                max_inference_batch_size=1,
                max_new_tokens=512,
            )
        return asr_models[model_name]

    def load_translator():
        nonlocal translator, tokenizer
        if translator is None:
            tokenizer = AutoTokenizer.from_pretrained(settings.translator)
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            translator = AutoModelForCausalLM.from_pretrained(
                settings.translator,
                device_map="cuda:0",
                quantization_config=quantization,
                attn_implementation="sdpa",
            )
        return tokenizer, translator

    def translate(messages: list[dict[str, str]]) -> str:
        local_tokenizer, local_translator = load_translator()
        prompt = local_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _remove_unsupported_generation_inputs(
            local_tokenizer(prompt, return_tensors="pt").to(local_translator.device)
        )
        with torch.inference_mode():
            generated = local_translator.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.02,
            )
        return local_tokenizer.decode(
            generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()


    try:
        for model_name in _startup_asr_models(settings):
            load_asr(model_name)
        load_translator()
        output_queue.put(InferenceResult(
            kind="ready", segment_id="", revision=0, final=False, text="",
            model_name=settings.primary_asr, audio_start_ms=0, audio_end_ms=0,
            elapsed_seconds=0, queue_seconds=0, gpu_memory_mb=gpu_memory(),
        ))
    except Exception:
        output_queue.put(InferenceResult(
            kind="error", segment_id="", revision=0, final=False, text="",
            model_name=settings.primary_asr, audio_start_ms=0, audio_end_ms=0,
            elapsed_seconds=0, queue_seconds=0, gpu_memory_mb=gpu_memory(),
            error=traceback.format_exc(),
        ))
        return

    while True:
        if not pending:
            incoming = input_queue.get()
            if incoming is None:
                break
            heapq.heappush(pending, incoming)
        while True:
            try:
                incoming = input_queue.get_nowait()
            except queue.Empty:
                break
            if incoming is None:
                return
            heapq.heappush(pending, incoming)
        pending, superseded = _coalesce_pending(pending)
        for skipped in superseded:
            output_queue.put(InferenceResult(
                kind="superseded",
                segment_id=skipped.segment_id,
                revision=skipped.revision,
                final=skipped.final,
                text="",
                model_name=skipped.model_name,
                audio_start_ms=skipped.audio_start_ms,
                audio_end_ms=skipped.audio_end_ms,
                elapsed_seconds=0,
                queue_seconds=time.monotonic() - skipped.submitted_monotonic,
                gpu_memory_mb=gpu_memory(),
            ))
        task = heapq.heappop(pending)
        started = time.perf_counter()
        queued = time.monotonic() - task.submitted_monotonic
        try:
            if task.kind == "asr":
                assert task.audio is not None
                model = load_asr(task.model_name)
                response = model.transcribe(
                    audio=(task.audio, settings.sample_rate), language="Chinese"
                )[0]
                text = response.text.strip()
            else:
                system = (
                    "Translate Mandarin Chinese meeting captions into faithful, natural English. "
                    "Preserve every technical detail, number, product name, decision, question, and uncertainty. "
                    f"Use this verified project glossary for exact or phonetic variants: {settings.prompt_terms}. "
                    "Do not summarize or add explanations. Return only the English translation."
                )
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": task.text},
                ]
                text = translate(messages)
                if _contains_cjk(text):
                    messages[0]["content"] = (
                        "Translate the entire Mandarin Chinese meeting caption into faithful, "
                        "natural English. Output English only: translate or, for unknown names, "
                        "romanize every Chinese character. Never copy Chinese characters into "
                        "the answer. Do not summarize or add explanations."
                    )
                    text = translate(messages)
            output_queue.put(InferenceResult(
                kind=task.kind,
                segment_id=task.segment_id,
                revision=task.revision,
                final=task.final,
                text=text,
                model_name=task.model_name or settings.translator,
                audio_start_ms=task.audio_start_ms,
                audio_end_ms=task.audio_end_ms,
                elapsed_seconds=time.perf_counter() - started,
                queue_seconds=queued,
                gpu_memory_mb=gpu_memory(),
            ))
        except Exception:
            output_queue.put(InferenceResult(
                kind="error",
                segment_id=task.segment_id,
                revision=task.revision,
                final=task.final,
                text="",
                model_name=task.model_name,
                audio_start_ms=task.audio_start_ms,
                audio_end_ms=task.audio_end_ms,
                elapsed_seconds=time.perf_counter() - started,
                queue_seconds=queued,
                gpu_memory_mb=gpu_memory(),
                error=traceback.format_exc(),
            ))
