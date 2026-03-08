---
name: langsmith-fetch
description: Debug LangChain and LangGraph agents by fetching execution traces from LangSmith Studio. Use when debugging agent behavior, investigating errors, analyzing tool calls, checking memory operations, or examining agent performance. Automatically fetches recent traces and analyzes execution patterns. Requires langsmith-fetch CLI installed.
---

# LangSmith Fetch – Agent Debugging Skill

This skill helps you debug LangChain and LangGraph agents by fetching and analyzing traces from LangSmith Studio directly in your terminal, with Copilot guiding you through the commands and interpreting the results.

## When to Use This Skill

Activate this skill when the user mentions any of the following:
- 🐛 "Debug my agent" or "What went wrong?"
- 🔍 "Show me recent traces" or "What happened?"
- ❌ "Check for errors" or "Why did it fail?"
- 💾 "Analyze memory operations" or "Check LTM"
- 📊 "Review agent performance" or "Check token usage"
- 🔧 "What tools were called?" or "Show execution flow"

## Prerequisites (User Must Have)

### 1. Install langsmith-fetch
```bash
pip install langsmith-fetch
```

### 2. Set Environment Variables
```bash
export LANGSMITH_API_KEY="your_langsmith_api_key"
export LANGSMITH_PROJECT="your_project_name"
```

**Verify setup:**
```bash
echo $LANGSMITH_API_KEY
echo $LANGSMITH_PROJECT
```

## Core Workflows – How to Guide the User

### Workflow 1: Quick Debug Recent Activity

**When the user asks:** "What just happened?" or "Debug my agent"

**Your response should:**
1. Suggest running:
   ```bash
   langsmith-fetch traces --last-n-minutes 5 --limit 5 --format pretty
   ```
2. Ask the user to run the command and paste the output.
3. Analyze the output and provide a summary:
   - ✅ Number of traces found
   - ⚠️ Any errors or failures
   - 🛠️ Tools that were called
   - ⏱️ Execution times
   - 💰 Token usage

**Example analysis format:**
```
Found 3 traces in the last 5 minutes:

Trace 1: ✅ Success
- Agent: memento
- Tools: recall_memories, create_entities
- Duration: 2.3s
- Tokens: 1,245

Trace 2: ❌ Error
- Agent: cypher
- Error: "Neo4j connection timeout"
- Duration: 15.1s
- Failed at: search_nodes tool

Trace 3: ✅ Success
- Agent: memento
- Tools: store_memory
- Duration: 1.8s
- Tokens: 892

💡 Issue found: Trace 2 failed due to Neo4j timeout. Recommend checking database connection.
```

---

### Workflow 2: Deep Dive Specific Trace

**When the user provides:** A trace ID or says "investigate that error"

**Your response should:**
1. Ask them to run:
   ```bash
   langsmith-fetch trace <trace-id> --format json
   ```
2. Once they paste the JSON, analyze and report:
   - 🎯 What the agent was trying to do
   - 🛠️ Tools called (in order)
   - ✅ Tool results (success/failure)
   - ❌ Error messages (if any)
   - 💡 Root cause analysis
   - 🔧 Suggested fix

**Example analysis:**
```
Deep Dive Analysis - Trace abc123

Goal: User asked "Find all projects in Neo4j"

Execution Flow:
1. ✅ search_nodes(query: "projects")
   → Found 24 nodes

2. ❌ get_node_details(node_id: "proj_123")
   → Error: "Node not found"
   → This is the failure point

3. ⏹️ Execution stopped

Root Cause:
The search_nodes tool returned node IDs that no longer exist in the database,
possibly due to recent deletions.

Suggested Fix:
1. Add error handling in get_node_details tool
2. Filter deleted nodes in search results
3. Update cache invalidation strategy

Token Usage: 1,842 tokens ($0.0276)
Execution Time: 8.7 seconds
```

---

### Workflow 3: Export Debug Session

**When the user says:** "Save this session" or "Export traces"

**Your response should provide commands:**
```bash
SESSION_DIR="langsmith-debug/session-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SESSION_DIR"
langsmith-fetch traces "$SESSION_DIR/traces" --last-n-minutes 30 --limit 50 --include-metadata
langsmith-fetch threads "$SESSION_DIR/threads" --limit 20
```

Then summarize the exported data or create a report from the files.

---

### Workflow 4: Error Detection

**When the user asks:** "Show me errors" or "What's failing?"

**Your response should suggest:**
```bash
langsmith-fetch traces --last-n-minutes 30 --limit 50 --format json > recent-traces.json
grep -i "error\|failed\|exception" recent-traces.json
```

Then analyze the output and report:
- 📊 Total errors found
- ❌ Error types and frequency
- 🕐 When errors occurred
- 🎯 Which agents/tools failed
- 💡 Common patterns

**Example analysis:**
```
Error Analysis - Last 30 Minutes

Total Traces: 50
Failed Traces: 7 (14% failure rate)

Error Breakdown:
1. Neo4j Connection Timeout (4 occurrences)
   - Agent: cypher
   - Tool: search_nodes
   - First occurred: 14:32
   - Last occurred: 14:45
   - Pattern: Happens during peak load

2. Memory Store Failed (2 occurrences)
   - Agent: memento
   - Tool: store_memory
   - Error: "Pinecone rate limit exceeded"
   - Occurred: 14:38, 14:41

3. Tool Not Found (1 occurrence)
   - Agent: sqlcrm
   - Attempted tool: "export_report" (doesn't exist)
   - Occurred: 14:35

💡 Recommendations:
1. Add retry logic for Neo4j timeouts
2. Implement rate limiting for Pinecone
3. Fix sqlcrm tool configuration
```

---

## Common Use Cases – Guidance to Provide

### Use Case 1: "Agent Not Responding"

**When the user says:** "My agent isn't doing anything"

**Your response should guide them through:**
1. Check if traces exist:
   ```bash
   langsmith-fetch traces --last-n-minutes 5 --limit 5
   ```
2. **If NO traces found:**
   - Verify tracing is enabled (`LANGCHAIN_TRACING_V2=true`)
   - Check API key and project name
   - Ensure the agent actually ran
3. **If traces found:**
   - Review for errors
   - Check execution time
   - Verify tool calls completed

### Use Case 2: "Wrong Tool Called"

**When the user says:** "Why did it use the wrong tool?"

**Your response should:**
- Ask for the trace ID and run deep dive
- Review available tools at execution time
- Suggest checking tool descriptions/instructions
- Recommend prompt or tool config improvements

### Use Case 3: "Memory Not Working"

**When the user says:** "Agent doesn't remember things"

**Your response should suggest:**
```bash
langsmith-fetch traces --last-n-minutes 10 --limit 20 --format raw | grep -i "memory\|recall\|store"
```
Then help analyze:
- Were memory tools called?
- Did recall return results?
- Were memories stored?
- Are retrieved memories being used?

### Use Case 4: "Performance Issues"

**When the user says:** "Agent is too slow"

**Your response should propose:**
```bash
langsmith-fetch traces ./perf-analysis --last-n-minutes 30 --limit 50 --include-metadata
```
Then help analyze:
- Execution time per trace
- Tool call latencies
- Token usage
- Number of iterations
- Slowest operations
- Identify bottlenecks and suggest optimizations

---

## Output Format Guide

When the user pastes command output, interpret the format:
- **Pretty format** – for quick visual inspection
- **JSON format** – parse and highlight issues
- **Raw format** – use for pattern matching

You can also help reformat output for better readability.

---

## Advanced Features – You Can Suggest

- **Time-based filtering:** `--after "2025-12-24T13:00:00Z"` or `--last-n-minutes 60`
- **Include metadata:** `--include-metadata`
- **Concurrent fetching:** `--concurrent 10` for faster exports

---

## Troubleshooting – How to Diagnose

When the user encounters issues like "No traces found" or "Project not found", guide them:

```
It looks like no traces were found. Let's troubleshoot:

1. Run `echo $LANGSMITH_API_KEY` to ensure it's set.
2. Run `echo $LANGSMITH_PROJECT` to verify the project name.
3. Check if tracing is enabled in your code: LANGCHAIN_TRACING_V2=true
4. Try a longer timeframe: `langsmith-fetch traces --last-n-minutes 1440 --limit 50`
5. If still empty, verify your agent actually ran during that period.
```

---

## Best Practices – Remind the User

- Run regular health checks: `langsmith-fetch traces --last-n-minutes 5 --limit 5`
- Organize exports in folders like `langsmith-debug/sessions/`
- Document findings: when a bug is found, export the trace and save with notes
- Integrate checks into development workflow (e.g., pre-commit hooks)

---

## Quick Reference Commands

Provide these on demand:
- Fetch recent traces: `langsmith-fetch traces --last-n-minutes 5 --limit 5 --format pretty`
- Get a specific trace: `langsmith-fetch trace <trace-id> --format pretty`
- Export a session: `langsmith-fetch traces ./debug-session --last-n-minutes 30 --limit 50`
- Find errors: `langsmith-fetch traces --last-n-minutes 30 --limit 50 --format raw | grep -i error`
- Include metadata: `langsmith-fetch traces --limit 10 --include-metadata`

---

## Notes for the AI (Your Own Instructions)

- Always check if the user has installed `langsmith-fetch` and set environment variables before suggesting commands. If not, guide them through those steps.
- Suggest commands, but let the user run them and paste output.
- When analyzing output, focus on actionable insights: errors, performance, tool usage.
- Provide clear, concise summaries and specific recommendations.
- If commands fail, help troubleshoot the setup.
- Remember that you are acting as a debugging assistant specialized in LangSmith traces.

---

**Version:** 0.1.0  
**Adapted from:** [langsmith-fetch-skill for Claude](https://github.com/OthmanAdi/langsmith-fetch-skill)
