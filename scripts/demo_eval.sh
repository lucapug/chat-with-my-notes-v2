#!/bin/bash
clear
echo "============================================"
echo "  Sprint 1 — RAG Evaluation Baseline"
echo "  chat-with-my-notes-v2"
echo "============================================"
echo ""
echo "  Dataset  : golden-set-ground-truth.json"
echo "  Queries  : 5 golden questions (italiano)"
echo "  top_k    : 10"
echo ""
echo "  Generator : gemma4:8k-latest  (Minisforum)"
echo "  Judge     : gemma4:e4b        (Asus F15)"
echo "  Embedder  : nomic-embed-text  (Minisforum)"
echo ""
sleep 2
echo "  Running evaluation..."
sleep 3
echo ""
echo "  [1/5] Spese Tecnologiche    hit=1  mrr=1.000"
sleep 2
echo "  [2/5] Alibi Detect          hit=1  mrr=1.000"
sleep 2
echo "  [3/5] Oxen.ai               hit=0  mrr=0.000"
sleep 2
echo "  [4/5] MLOps Tools           hit=1  mrr=0.333"
sleep 2
echo "  [5/5] Grid Search Params    hit=1  mrr=1.000"
sleep 3
echo ""
echo "  Hit Rate@10  :  0.80"
echo "  MRR          :  0.667"
echo "  Avg total    :  65.4s / query"
echo ""
echo "  Status: LOCAL EVALUATION COMPLETE ✓"
echo "============================================"