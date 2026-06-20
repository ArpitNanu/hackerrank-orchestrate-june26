import argparse
import sys
import traceback

from claim_review.csv_io import read_claims, read_user_history, read_requirements, write_output
from claim_review.pipeline import process_claim

def main():
    parser = argparse.ArgumentParser(description="HackerRank Orchestrate - Claim Review Pipeline")
    parser.add_argument("--claims", required=True, help="Path to claims.csv")
    parser.add_argument("--history", required=True, help="Path to user_history.csv")
    parser.add_argument("--requirements", required=True, help="Path to evidence_requirements.csv")
    parser.add_argument("--output", required=True, help="Path to write output.csv")
    
    args = parser.parse_args()
    
    try:
        print("Loading datasets...")
        # 1. Load claims.csv
        claims = read_claims(args.claims)
        
        # 2. Load user_history.csv
        user_history_map = read_user_history(args.history)
        
        # 3. Load evidence_requirements.csv
        requirements = read_requirements(args.requirements)
        
        # Inject loaded requirements into the resolver's cache so it respects the CLI argument
        import claim_review.requirements_resolver as rr
        rr._CACHED_REQUIREMENTS = requirements
        
        output_rows = []
        
        print(f"Loaded {len(claims)} claims. Beginning pipeline execution...")
        
        # 4. Run pipeline for every claim
        for idx, row in enumerate(claims):
            user_id = row.get("user_id", "")
            user_claim = row.get("user_claim", "")
            claim_object = row.get("claim_object", "")
            image_paths_str = row.get("image_paths", "")
            
            print(f"[{idx+1}/{len(claims)}] Processing claim for user {user_id}...")
            
            # Extract paths from the semicolon-separated string
            image_paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]
            
            try:
                # Look up user history for this specific user
                raw_history = user_history_map.get(user_id, {})
                
                # Parse into Pydantic model so structured data is available
                from claim_review.schemas import UserHistoryRow
                user_history = UserHistoryRow(
                    past_claim_count=int(raw_history.get("past_claim_count", 0)),
                    accept_claim=int(raw_history.get("accept_claim", 0)),
                    manual_review_claim=int(raw_history.get("manual_review_claim", 0)),
                    rejected_claim=int(raw_history.get("rejected_claim", 0)),
                    last_90_days_claim_count=int(raw_history.get("last_90_days_claim_count", 0)),
                    history_flags=raw_history.get("history_flags", ""),
                    history_summary=raw_history.get("history_summary", "")
                )
                
                # The pipeline acts as the central orchestrator. 
                # user_history is passed directly so risk_assessor can evaluate it.
                # Requirements are pre-loaded into the resolver cache.
                final_output = process_claim(
                    user_id=user_id,
                    user_claim=user_claim,
                    claim_object=claim_object,
                    image_paths=image_paths,
                    user_history=user_history,
                    evidence_requirement=None  # Can be resolved internally by the pipeline
                )
                output_rows.append(final_output)
            except Exception as e:
                print(f"  [Error] Failed to process claim for user {user_id}: {e}")
                traceback.print_exc()
                # Continue processing the rest of the claims gracefully
                
        # 5. Write output.csv
        print(f"Writing {len(output_rows)} rows to {args.output}...")
        write_output(args.output, output_rows)
        print("Pipeline execution completed successfully.")
        
    except Exception as e:
        print(f"[Critical Error] Pipeline execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
