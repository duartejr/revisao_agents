import glob
import os

from .agents.reference_formatter_agent import (
    run_reference_formatter_agent,
)
from .config import print_runtime_config_summary, validate_runtime_config
from .core.schemas.writer_config import WriterConfig
from .hitl import run_hitl_loop
from .state import ReviewState, TechnicalWriterState
from .utils.vector_utils.pdf_ingestor import ingest_pdf_folder
from .workflows import build_academic_workflow, build_technical_workflow
from .workflows.technical_writing_workflow import build_technical_writing_workflow


def main():
    # Ensure output directories exist
    os.makedirs("plans", exist_ok=True)
    os.makedirs("reviews", exist_ok=True)

    print_runtime_config_summary()
    startup_issues = validate_runtime_config(strict=False)
    if startup_issues:
        print("⚠️  Configuration warnings detected:")
        for issue in startup_issues:
            print(f"   - {issue}")
        print("   (The flow may fail in options that require these integrations.)\n")

    print("\n" + "=" * 70)
    print("REVIEW PLANNING AGENT")
    print("=" * 70)
    print("\nOptions:")
    print("  [1] Plan Academic Review (narrative)")
    print("  [2] Plan Technical Review (chapter)")
    print("  [3] Execute Writing from Existing Plan (Technical or Academic)")
    print("  [4] Index Local PDFs → vectorize and save to MongoDB")
    print("  [5] Format References (ABNT, APA, IEEE, etc.) from YAML/JSON file")
    choice = input("\nChoose [1/2/3/4/5]: ").strip()

    if choice == "4":
        print("\n" + "=" * 70)
        print("INDEX LOCAL PDFs")
        print("=" * 70)
        folder = input("\nPath to folder with PDFs: ").strip()
        if not folder:
            print("❌ Empty path.")
            return
        folder = os.path.expanduser(folder)
        if not os.path.isdir(folder):
            print(f"❌ Folder not found: {folder}")
            return
        result = ingest_pdf_folder(folder)
        print("\n" + "=" * 70)
        print("INDEXING RESULT")
        print("=" * 70)
        print(f"  ✅ New PDFs indexed : {result['indexed']}")
        print(f"  ⏭️  Already in DB     : {result['already']}")
        print(f"  ⚠️  Insufficient text : {result['skipped']}")
        print(f"  ❌ Reading errors     : {result['errors']}")
        print(f"  📦 Chunks inserted    : {result['total_chunks']}")
        print("=" * 70)
        return

    if choice == "5":
        run_reference_formatter_agent()
        return

    if choice == "3":
        # --- Writing mode sub-menu ---
        print("\n" + "-" * 70)
        print("WRITING STYLE:")
        print("  [a] Technical section — didactic chapter (web search + MongoDB)")
        print("  [b] Academic — narrative literature review (corpus-first)")
        select_mode = input("\nChoose [a/b, default=a]: ").strip().lower() or "a"
        if select_mode == "b":
            glob_pattern_primary = "plans/plano_revisao_*.md"
            mode_label = "ACADEMIC"
        else:
            glob_pattern_primary = "plans/plano_revisao_tecnica_*.md"
            mode_label = "TECHNICAL"

        # --- Language selection ---
        print("\n" + "-" * 70)
        print("REVIEW LANGUAGE:")
        print("  [pt] Portuguese (standard)")
        print("  [en] English")
        lang_opt = input("\nChoose [pt/en, default=pt]: ").strip().lower() or "pt"
        if lang_opt not in ("pt", "en"):
            lang_opt = "pt"
        if select_mode == "b":
            writer_config = WriterConfig.academic(language=lang_opt)
        else:
            writer_config = WriterConfig.technical(language=lang_opt)
        print(f"   ✔  Language: {'Portuguese (pt-BR)' if lang_opt == 'pt' else 'English'}")

        # --- Minimum distinct sources per section ---
        default_min = 4 if select_mode == "b" else 0
        print("\n" + "-" * 70)
        print("MINIMUM NUMBER OF DISTINCT SOURCES PER SECTION:")
        print(f"  (default = {default_min}; 0 = no restriction)")
        min_src_input = input(f"\nMinimum sources per section [{default_min}]: ").strip()
        try:
            min_src = int(min_src_input) if min_src_input else default_min
        except ValueError:
            min_src = default_min
        if min_src < 0:
            min_src = 0
        writer_config.min_sources_per_section = min_src
        print(f"   ✔  Minimum sources/section: {min_src}")

        # --- Tavily search option ---
        print("\n" + "-" * 70)
        print("Do you want to enable web/image search via Tavily?")
        print("  [y] Yes (web and image search)")
        print("  [n] No (local corpus only)")
        tavily_opt = input("\nEnable Tavily? [y/N]: ").strip().lower() or "n"
        tavily_enabled = tavily_opt == "y"

        print("\n" + "=" * 70)
        print(f"WRITING EXECUTION {mode_label}")
        print("=" * 70)

        # --- Find plan files ---
        planos = sorted(glob.glob(glob_pattern_primary))
        if not planos:
            planos = sorted(glob.glob("plans/review_plan*.md"))  # broader fallback
        if not planos:
            planos = sorted(glob.glob("review_plan*.md"))  # root fallback
        if planos:
            print("\nPlans found:")
            for i, p in enumerate(planos, 1):
                print(f"  [{i}] {p}")
            idx = input(f"\nChoose [1-{len(planos)} or path]: ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(planos):
                plan_path = planos[int(idx) - 1]
            else:
                plan_path = idx
        else:
            plan_path = input("\nPlan path (.md): ").strip()

        if not os.path.exists(plan_path):
            print(f"❌ File not found: {plan_path}")
            return

        state_init: TechnicalWriterState = {
            "theme": "",
            "plan_summary": "",
            "sections": [],
            "plan_path": plan_path,
            "written_sections": [],
            "refs_urls": [],
            "refs_images": [],
            "cumulative_summary": "",
            "react_log": [],
            "verification_stats": [],
            "status": "starting",
            "writer_config": writer_config.to_dict(),
            "tavily_enabled": tavily_enabled,
        }
        app = build_technical_writing_workflow()
        try:
            for event in app.stream(state_init):
                node = list(event.keys())[0] if event else "?"
                if node != "__end__":
                    st = event.get(node, {}).get("status", "")
                    if st:
                        print(f"\n   ▶ [{node}] → {st}")
            print("\n" + "=" * 70)
            print("✅ WRITING COMPLETED")
            print("=" * 70)
        except KeyboardInterrupt:
            print("\nCancelled.")
        return

    # --- Planning (options 1 and 2) ---
    theme = input("\nReview theme: ").strip()
    if not theme:
        print("Empty theme.")
        return

    print("\n[1] Academic Review (literature narrative) — search in MongoDB corpus")
    print("[2] Technical Review   (detailed didactic chapter) — search on the internet")
    print("[3] Both")
    while True:
        e = choice if choice in ("1", "2", "3") else input("\nChoice [1/2/3]: ").strip()
        if e in ("1", "2", "3"):
            break
        choice = ""  # Clear to order again
        print("Enter 1, 2, or 3.")
    review_types = {
        "1": ["academic"],
        "2": ["technical"],
        "3": ["academic", "technical"],
    }[e]

    max_p = 3
    try:
        n = input("Refinement rounds per plan [default 3]: ").strip()
        if n.isdigit() and int(n) > 0:
            max_p = min(int(n), 6)
    except Exception:
        pass

    for review_type in review_types:
        label = "ACADEMIC" if review_type == "academic" else "TECHNICAL"
        print("\n" + "=" * 70)
        print(f"Starting: REVIEW {label} | {repr(theme)} | {max_p} rounds")
        print("-" * 70)

        state_init: ReviewState = {
            "theme": theme,
            "review_type": review_type,
            "relevant_chunks": [],
            "technical_snippets": [],
            "technical_urls": [],
            "current_plan": "",
            "interview_history": [],
            "questions_asked": 0,
            "max_questions": max_p,
            "final_plan": "",
            "final_plan_path": "",
            "status": "starting",
        }
        config = {"configurable": {"thread_id": f"review_{review_type}_{theme[:20]}"}}

        app = build_academic_workflow() if review_type == "academic" else build_technical_workflow()

        try:
            run_hitl_loop(app, config, state_init)
        except KeyboardInterrupt:
            print("\nCanceled.")
            break
        except Exception as ex:
            import traceback

            print("\nError:", str(ex))
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Planning session completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
