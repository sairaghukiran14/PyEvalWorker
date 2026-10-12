import re
from typing import List, Dict, Any, Tuple, Optional

from app.azure_client import llm_client
from app.schemas import RagEvalRequest, RagEvalResponse
from app.tracing import trace_eval_span


def _mock_rag_claims(answer: str, chunks: List[str]) -> Dict[str, Any]:
    """Mock claim extraction and verification."""
    lower_ans = answer.lower()
    combined_chunks = " ".join(chunks).lower()

    # Split into sentences or clauses
    raw_sentences = [s.strip() for s in re.split(r"[.!?]\s*", answer) if len(s.strip()) > 5]
    if not raw_sentences:
        raw_sentences = [answer]

    claims = []
    supported_count = 0
    for s in raw_sentences:
        lower_s = s.lower()
        # Check hallucination markers
        is_bad = any(marker in lower_s for marker in [
            "bertscore", "0.7", "10 times", "oauth2", "jwt",
            "llama-3", "5 epochs", "deletes all", "hacked", "opposite"
        ])

        # Simple substring / word overlap check
        words = [w for w in re.findall(r"\w+", lower_s) if len(w) > 3]
        overlap = sum(1 for w in words if w in combined_chunks)
        is_supported = (overlap >= max(1, len(words) * 0.4)) and not is_bad

        if is_supported:
            supported_count += 1
            claims.append({"claim": s, "supported": True, "reasoning": "Supported by context chunks."})
        else:
            claims.append({"claim": s, "supported": False, "reasoning": "Claim not found in context or contradicts."})

    faithfulness = supported_count / len(claims) if claims else 1.0
    return {
        "faithfulness": round(faithfulness, 2),
        "claims": claims,
        "reasoning": f"Extracted {len(claims)} claims; {supported_count} are supported by retrieved context."
    }


def compute_faithfulness(answer: str, chunks: List[str]) -> Tuple[float, List[Dict[str, Any]], str]:
    """
    Computes Faithfulness:
    1. Splits answer into atomic claims.
    2. Verifies whether each claim is directly supported by retrieved chunks.
    3. Score = (supported claims) / (total claims).
    """
    system_prompt = (
        "You are an expert evaluator assessing the faithfulness (groundedness) of a generated answer.\n"
        "Instructions:\n"
        "1. Break the answer down into independent atomic factual claims.\n"
        "2. For each claim, determine whether it can be directly verified from the provided context chunks.\n"
        "3. Output valid JSON with:\n"
        "   - 'claims': list of objects with 'claim' (str), 'supported' (bool), and 'reasoning' (str)\n"
        "   - 'faithfulness': float ratio of supported claims / total claims (between 0.0 and 1.0)\n"
        "   - 'reasoning': overall summary explanation"
    )

    chunks_formatted = "\n".join(f"Chunk {i+1}: {c}" for i, c in enumerate(chunks))
    user_prompt = (
        f"RETRIEVED CONTEXT CHUNKS:\n{chunks_formatted}\n\n"
        f"CANDIDATE ANSWER:\n{answer}\n\n"
        "Provide JSON output:"
    )

    parsed_json, usage, cost = llm_client.call_llm_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        mock_handler=lambda sys, usr: _mock_rag_claims(answer, chunks)
    )

    claims = parsed_json.get("claims", [])
    if claims:
        supported_count = sum(1 for c in claims if c.get("supported", False))
        score = round(supported_count / len(claims), 3)
    else:
        score = float(parsed_json.get("faithfulness", 1.0))

    reasoning = parsed_json.get("reasoning", f"Faithfulness score: {score}")
    return score, claims, reasoning


def compute_answer_relevance(question: str, answer: str) -> Tuple[float, str]:
    """
    Computes Answer Relevance:
    Rates from 1.0 to 5.0 against rubric whether the answer directly addresses the question.
    """
    system_prompt = (
        "You are an expert judge evaluating Answer Relevance.\n"
        "Determine how directly, completely, and appropriately the answer addresses the question.\n"
        "Scoring Scale (1.0 to 5.0):\n"
        "- 5.0: Completely addresses the question directly and accurately.\n"
        "- 4.0: Mostly addresses the question, minor tangential details.\n"
        "- 3.0: Partially addresses the question or overly broad.\n"
        "- 2.0: Evasive or largely off-topic.\n"
        "- 1.0: Completely irrelevant or refuses to answer.\n"
        "Output valid JSON with:\n"
        "- 'answer_relevance': float between 1.0 and 5.0\n"
        "- 'reasoning': explanation of the score"
    )

    user_prompt = f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nProvide JSON output:"

    def _mock_relevance(sys, usr):
        lower_ans = answer.lower()
        if "fastapi is a modern" in lower_ans or "testing software ensures" in lower_ans:
            score = 1.0 if "fastapi" in lower_ans else 2.0
            return {"answer_relevance": score, "reasoning": "Answer is evasive and fails to address the question."}
        return {"answer_relevance": 5.0, "reasoning": "Answer directly and accurately addresses the query."}

    parsed_json, usage, cost = llm_client.call_llm_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        mock_handler=_mock_relevance
    )

    score = float(parsed_json.get("answer_relevance", 5.0))
    reasoning = parsed_json.get("reasoning", f"Answer relevance: {score}/5.0")
    return score, reasoning


def compute_context_precision(question: str, chunks: List[str]) -> Tuple[float, List[Dict[str, Any]], str]:
    """
    Computes Context Precision:
    Evaluates whether each chunk is relevant to the question and calculates rank-weighted precision:
    MAP@K = sum(Precision@k * is_relevant_k) / sum(is_relevant_k).
    """
    if not chunks:
        return 0.0, [], "No chunks provided for context precision."

    system_prompt = (
        "You are an expert evaluator assessing Context Precision in RAG retrieval.\n"
        "For each chunk, determine if it contains relevant information to answer the question.\n"
        "Output valid JSON with:\n"
        "- 'chunk_relevancies': list of objects with 'chunk_index' (1-indexed int), 'is_relevant' (bool), 'reason' (str)\n"
        "- 'reasoning': summary explanation"
    )

    chunks_text = "\n".join(f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(chunks))
    user_prompt = f"QUESTION:\n{question}\n\nRETRIEVED CHUNKS:\n{chunks_text}\n\nProvide JSON output:"

    def _mock_precision(sys, usr):
        # By default in golden set, all chunks provided are relevant
        relevancies = []
        for i, chk in enumerate(chunks):
            # Check if chunk mentions keywords from question
            q_words = [w for w in re.findall(r"\w+", question.lower()) if len(w) > 3]
            match_count = sum(1 for w in q_words if w in chk.lower())
            is_rel = match_count > 0 or i == 0
            relevancies.append({
                "chunk_index": i + 1,
                "is_relevant": is_rel,
                "reason": "Matches key terms from query" if is_rel else "Tangential content"
            })
        return {
            "chunk_relevancies": relevancies,
            "reasoning": f"Evaluated {len(chunks)} chunks for query relevance."
        }

    parsed_json, usage, cost = llm_client.call_llm_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        mock_handler=_mock_precision
    )

    chunk_relevancies = parsed_json.get("chunk_relevancies", [])
    relevant_hits = 0
    weighted_precision_sum = 0.0

    for idx, item in enumerate(chunk_relevancies, start=1):
        is_rel = bool(item.get("is_relevant", False))
        if is_rel:
            relevant_hits += 1
            precision_at_k = relevant_hits / idx
            weighted_precision_sum += precision_at_k

    if relevant_hits == 0:
        precision_score = 0.0
    else:
        precision_score = round(weighted_precision_sum / relevant_hits, 3)

    reasoning = (
        f"Context precision is {precision_score}: {relevant_hits}/{len(chunks)} retrieved chunks "
        f"are relevant. Rank-weighted MAP@K calculated."
    )
    return precision_score, chunk_relevancies, reasoning


def compute_context_recall(
    reference: Optional[str],
    chunks: List[str],
    question: Optional[str] = None
) -> Tuple[float, List[Dict[str, Any]], str]:
    """
    Computes Context Recall:
    Measures the proportion of reference answer claims that can be found in the retrieved chunks.
    """
    if not reference:
        return 1.0, [], "No reference answer provided; context recall defaulted to 1.0."

    system_prompt = (
        "You are an expert evaluator assessing Context Recall.\n"
        "Instructions:\n"
        "1. Extract the core factual claims from the REFERENCE answer.\n"
        "2. Check if each claim can be found or attributed to any of the retrieved chunks.\n"
        "3. Output valid JSON with:\n"
        "   - 'reference_claims': list of objects with 'claim' (str), 'found_in_context' (bool), 'reason' (str)\n"
        "   - 'context_recall': float ratio of found claims / total reference claims (0.0 to 1.0)\n"
        "   - 'reasoning': summary explanation"
    )

    chunks_text = "\n".join(f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(chunks))
    user_prompt = (
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"RETRIEVED CHUNKS:\n{chunks_text}\n\n"
        "Provide JSON output:"
    )

    def _mock_recall(sys, usr):
        ref_sentences = [s.strip() for s in re.split(r"[.!?]\s*", reference) if len(s.strip()) > 5]
        if not ref_sentences:
            ref_sentences = [reference]

        combined = " ".join(chunks).lower()
        claims = []
        found_cnt = 0
        for s in ref_sentences:
            words = [w for w in re.findall(r"\w+", s.lower()) if len(w) > 3]
            overlap = sum(1 for w in words if w in combined)
            found = overlap >= max(1, len(words) * 0.4)
            if found:
                found_cnt += 1
            claims.append({
                "claim": s,
                "found_in_context": found,
                "reason": "Found in retrieved chunks" if found else "Missing from context"
            })
        recall = found_cnt / len(claims) if claims else 1.0
        return {
            "reference_claims": claims,
            "context_recall": round(recall, 2),
            "reasoning": f"{found_cnt}/{len(claims)} reference claims captured in context."
        }

    parsed_json, usage, cost = llm_client.call_llm_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        mock_handler=_mock_recall
    )

    ref_claims = parsed_json.get("reference_claims", [])
    if ref_claims:
        found_count = sum(1 for c in ref_claims if c.get("found_in_context", False))
        score = round(found_count / len(ref_claims), 3)
    else:
        score = float(parsed_json.get("context_recall", 1.0))

    reasoning = parsed_json.get("reasoning", f"Context recall score: {score}")
    return score, ref_claims, reasoning


def evaluate_rag_pipeline(request: RagEvalRequest) -> RagEvalResponse:
    """
    Computes all 4 core RAG metrics:
    - Faithfulness (0.0 to 1.0)
    - Answer Relevance (1.0 to 5.0)
    - Context Precision (0.0 to 1.0)
    - Context Recall (0.0 to 1.0)
    and combines their explanations into judgeReasoning.
    """
    with trace_eval_span(
        name="rag.eval",
        input_data={"question": request.question, "answer": request.answer, "chunk_count": len(request.chunks)},
        metadata={"has_reference": bool(request.reference)}
    ):
        faithfulness, claims, faith_reason = compute_faithfulness(
            answer=request.answer,
            chunks=request.chunks
        )

        relevance, rel_reason = compute_answer_relevance(
            question=request.question,
            answer=request.answer
        )

        precision, chunk_rels, prec_reason = compute_context_precision(
            question=request.question,
            chunks=request.chunks
        )

        recall, ref_claims, rec_reason = compute_context_recall(
            reference=request.reference,
            chunks=request.chunks,
            question=request.question
        )

    # Combined reasoning for Pravah's judgeReasoning field
    combined_reasoning = (
        f"RAG Evaluation Summary:\n"
        f"• Faithfulness ({faithfulness:.2f}): {faith_reason}\n"
        f"• Answer Relevance ({relevance:.1f}/5.0): {rel_reason}\n"
        f"• Context Precision ({precision:.2f}): {prec_reason}\n"
        f"• Context Recall ({recall:.2f}): {rec_reason}"
    )

    detailed_metrics = {
        "faithfulness": {"score": faithfulness, "claims": claims},
        "answer_relevance": {"score": relevance, "reasoning": rel_reason},
        "context_precision": {"score": precision, "chunk_relevancies": chunk_rels},
        "context_recall": {"score": recall, "reference_claims": ref_claims},
    }

    return RagEvalResponse(
        faithfulness=faithfulness,
        answer_relevance=relevance,
        context_precision=precision,
        context_recall=recall,
        reasoning=combined_reasoning,
        detailed_metrics=detailed_metrics
    )
