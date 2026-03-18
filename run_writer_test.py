#!/usr/bin/env python3
"""
Non-interactive test runner for the technical writer agent.

Usage (from revisao_agent/ directory):
    python run_writer_test.py

Monitors all phases, logs progress to stdout, and runs post-execution
quality checks on the generated review file.
"""

import os
import sys
import time
import re
import glob
from datetime import datetime
from pathlib import Path

# ── Resolve package path ─────────────────────────────────────────────────────
AGENT_DIR = Path(__file__).parent
SRC_DIR = AGENT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
os.chdir(AGENT_DIR)          # reviews/ and plans/ are relative to this dir

PLAN_FILE = str(AGENT_DIR / "plans" / "plano_revisao_tecnica_capítulo_test.md")

# ── Pre-flight checks ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PRE-FLIGHT: Import & parse check")
print("=" * 70)

# 1. Import check
try:
    from revisao_agents.nodes.technical_writing import (
        parsear_plano_node, escrever_secoes_node, consolidar_node,
        _fase_pensamento, _juiz_paragrafo_melhorado,
        _buscar_conteudo_complementar,
    )
    from revisao_agents.utils.llm_utils.prompt_loader import load_prompt
    from revisao_agents.workflows.technical_writing_workflow import build_workflow
    print("✅ All imports OK")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# 2. YAML prompt resolution check
try:
    p = load_prompt("technical_writing/fase_pensamento",
                    tema="test", titulo="test", cont_esp="test", recursos="test")
    print(f"✅ fase_pensamento.yaml resolved ({len(p.text)} chars, temp={p.temperature})")
except Exception as e:
    print(f"❌ fase_pensamento.yaml failed: {e}")
    sys.exit(1)

try:
    p = load_prompt("technical_writing/writer_judge",
                    paragrafo_limpo="test", titulo_secao="test", fontes="test")
    print(f"✅ writer_judge.yaml resolved ({len(p.text)} chars)")
except Exception as e:
    print(f"❌ writer_judge.yaml failed: {e}")
    sys.exit(1)

try:
    p = load_prompt("technical_writing/busca_complementar",
                    titulo_secao="test", conteudo_esperado="test")
    print(f"✅ busca_complementar.yaml resolved ({len(p.text)} chars, temp={p.temperature})")
except Exception as e:
    print(f"❌ busca_complementar.yaml failed: {e}")
    sys.exit(1)

# 3. Plan file check
if not os.path.exists(PLAN_FILE):
    print(f"❌ Plan file not found: {PLAN_FILE}")
    sys.exit(1)
print(f"✅ Plan file found: {Path(PLAN_FILE).name}")

# 4. Parse plan sections
try:
    from revisao_agents.utils.file_utils.helpers import parse_technical_plan
    with open(PLAN_FILE, "r", encoding="utf-8") as f:
        plan_text = f.read()
    tema, resumo, secoes = parse_technical_plan(plan_text)
    print(f"✅ Plan parsed: tema='{tema[:60]}' | {len(secoes)} section(s)")
    for s in secoes:
        print(f"   [{s['index']+1}] {s['title']}")
        print(f"        content: {s['expected_content'][:80]}")
        print(f"        resources: {s['resources'][:80]}")
except Exception as e:
    print(f"❌ Plan parse failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

if not secoes:
    print("❌ No sections found — check plan file format")
    sys.exit(1)

# ── Full Run ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"STARTING WRITER — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Plan: {Path(PLAN_FILE).name}")
print(f"Sections: {len(secoes)}")
print("=" * 70)

from revisao_agents.state import TechnicalWriterState

state_init: TechnicalWriterState = {
    "theme": "",
    "plan_summary": "",
    "sections": [],
    "plan_path": PLAN_FILE,
    "written_sections": [],
    "refs_urls": [],
    "refs_images": [],
    "cumulative_summary": "",
    "react_log": [],
    "verification_stats": [],
    "status": "starting",
}

os.makedirs("reviews", exist_ok=True)

run_start = time.time()
final_state = {}
last_status = {}

app = build_workflow()
try:
    for event in app.stream(state_init):
        node = list(event.keys())[0] if event else "?"
        if node == "__end__":
            continue
        node_state = event.get(node, {})
        st = node_state.get("status", "")
        if st:
            elapsed = time.time() - run_start
            print(f"\n   ▶ [{node}] → {st}  ({elapsed:.0f}s elapsed)")
        last_status = node_state
        final_state.update(node_state)
except KeyboardInterrupt:
    print("\n⚠️  Run cancelled by user.")
    sys.exit(1)
except Exception as e:
    import traceback
    print(f"\n❌ Writer failed: {e}")
    traceback.print_exc()
    sys.exit(1)

elapsed_total = time.time() - run_start
print(f"\n{'='*70}")
print(f"✅ WRITER COMPLETED — {elapsed_total:.0f}s total")
print(f"{'='*70}")

# ── Quality Checks ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("QUALITY CHECKS")
print("=" * 70)

# Find output file
md_files = sorted(glob.glob("reviews/revisao_tecnica_*.md"), key=os.path.getmtime, reverse=True)
log_files = sorted(glob.glob("reviews/revisao_tecnica_*.log"), key=os.path.getmtime, reverse=True)

if not md_files:
    print("❌ No output .md file found in reviews/")
    sys.exit(1)

out_file = md_files[0]
print(f"\n📄 Output: {out_file}")
file_size = os.path.getsize(out_file)
print(f"   Size: {file_size:,} bytes")

with open(out_file, "r", encoding="utf-8") as f:
    md_content = f.read()

n_lines = md_content.count("\n")
n_paragraphs = len([p for p in md_content.split("\n\n") if p.strip() and not p.strip().startswith("#")])

checks_passed = 0
checks_total = 0

def check(label, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    status = "✅" if condition else "❌"
    if condition:
        checks_passed += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"   {status} {label}{suffix}")

check("File size > 5KB", file_size > 5000, f"{file_size:,} bytes")
check("File has > 50 lines", n_lines > 50, f"{n_lines} lines")
check("Contains expected section (5.0 Aplicação)", "5.0 Aplicação" in md_content or "Aplicação em Hidrologia" in md_content)
check("Has >= 4 text paragraphs", n_paragraphs >= 4, f"{n_paragraphs} paragraphs")
check("Has References block", "Referências desta seção" in md_content)
check("Has verification stats comment", "verificados" in md_content)
check("Has LaTeX equations ($$)", "$$" in md_content)
check("Has algorithm/code blocks", "```" in md_content)
check("Has intro section", "## Introdução" in md_content)
check("Has conclusion section", "## Conclusão" in md_content)
check("No stub markers", "Simplified:" not in md_content and "Full implementation" not in md_content)
check("Verification blockquote present", "Verificação por parágrafo" in md_content)

# URL check in references
refs_match = re.findall(r"### Referências desta seção\n\n(.*?)(?:\n\n|$)", md_content, re.DOTALL)
has_urls = any(re.search(r"https?://", block) for block in refs_match) if refs_match else False
check("References contain URLs", has_urls)

# Paragraph density
section_match = re.search(r"## 5\.0 Aplicação.*?(?=\n##\s|\Z)", md_content, re.DOTALL)
if section_match:
    section_text = section_match.group(0)
    section_paragraphs = len([p for p in section_text.split("\n\n") if len(p.strip()) > 100])
    check("Section 5.0 has >= 4 substantial paragraphs", section_paragraphs >= 4, f"{section_paragraphs} found")
else:
    check("Section 5.0 found", False, "section not found in output")

print(f"\n{'─'*70}")
print(f"   Passed: {checks_passed}/{checks_total} checks")

# Log file check
if log_files:
    log_file = log_files[0]
    log_size = os.path.getsize(log_file)
    print(f"\n📋 Log: {log_file}  ({log_size:,} bytes)")
    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.read()
    log_checks = [
        ("Log has APROVADO entries", "APROVADO" in log_content),
        ("Log has per-section stats", "verificados" in log_content),
        ("Log has audit header", "REACT AUDIT LOG" in log_content),
    ]
    for label, cond in log_checks:
        checks_total += 1
        status = "✅" if cond else "❌"
        if cond: checks_passed += 1
        print(f"   {status} {label}")
else:
    print("❌ No log file found")

# Stats from final state
stats = final_state.get("verification_stats", [])
if stats:
    print(f"\n📊 Verification stats:")
    for s in stats:
        t = s.get("total", 0)
        aprov = s.get("aprovados", 0)
        ajust = s.get("ajustados", 0)
        corr = s.get("corrigidos", 0)
        anchors = s.get("anchors_usadas", 0)
        taxa = ((aprov + ajust) / t * 100) if t > 0 else 0
        print(f"   [{s.get('secao', '?')[:50]}]")
        print(f"     {aprov+ajust}/{t} verified ({taxa:.0f}%) | "
              f"✅{aprov} 🔵{ajust} 🔧{corr} | 🎯{anchors} anchors used")

print(f"\n{'='*70}")
print(f"FINAL: {checks_passed}/{checks_total} checks passed | "
      f"Runtime: {elapsed_total:.0f}s")
print(f"Output: {out_file}")
print(f"{'='*70}\n")

sys.exit(0 if checks_passed >= checks_total * 0.75 else 1)
