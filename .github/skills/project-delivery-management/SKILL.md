---
name: project-delivery-management
description: "Coordinates project delivery using roadmap planning, Jira-style sprint breakdown, risk and dependency control, and evidence-based status reporting from reports. Use for weekly sprint planning, backlog grooming, progress tracking, stakeholder updates, and reprioritization decisions."
argument-hint: "Plan or track this project delivery using roadmap and reports evidence"
user-invocable: true
---

# Project Delivery Management

## Outcome

Run this repository with a repeatable Data Product Manager and Delivery Manager workflow that converts roadmap goals into executable weekly sprints and validates execution through evidence files.

Primary sources in this project:
- roadmap plans: roadmap/
- execution evidence: reports/

## When to Use

Use this skill when you need to:
- convert roadmap phases into Jira-ready epics and stories
- plan a weekly sprint with strict acceptance criteria
- compare planned work versus delivered evidence
- produce risk-based reprioritization decisions
- publish stakeholder-ready status updates with delivery and quality KPIs

## Core Skill Map (Asana-inspired)

Apply these skill groups while planning and tracking delivery:

- Soft skills: collaboration, teamwork, communication, time management, leadership, organization, problem solving, critical thinking, adaptability, conflict resolution.
- Hard skills: project planning, project scoping, writing a project brief, kickoff facilitation, roadmapping, timeline mapping, task management.
- Technical skills: project tooling, timeline dependency mapping, Kanban flow, agile iteration, workload balancing, cost awareness, portfolio visibility, change management.

## Workflow

1. Intake and objective alignment.
2. Baseline evidence read.
3. Sprint design and Jira breakdown.
4. Dependency and risk design.
5. Execution sequencing.
6. Weekly status and KPI report.
7. Reprioritization and next-week handoff.

## Detailed Procedure

### 1) Intake and objective alignment

- Read the active roadmap file in roadmap/.
- Confirm target phase, target week, and desired output format.
- Produce a one-line sprint objective and 3-5 success criteria.

Output:
- Sprint objective
- Success criteria

### 2) Baseline evidence read

- Inspect reports/ for files that prove completed work.
- Build a planned-versus-evidence matrix:
  - planned artifact
  - evidence file path
  - status: done, partial, missing

Output:
- Evidence matrix

### 3) Sprint design and Jira breakdown

- Build one epic and story set for the target week.
- Each story must include mandatory fields:
  - ID
  - Summary
  - Description
  - Scope
  - Out of Scope
  - Subtasks
  - Acceptance Criteria
  - Estimate
  - Owner suggestion
  - Dependencies
  - Risks
  - Deliverables
  - Definition of Done

Output:
- Jira-ready sprint plan

### 4) Dependency and risk design

- List blockers and cross-task dependencies.
- Rate each risk with severity and impact.
- Attach mitigation and escalation trigger.

Output:
- Risk register

### 5) Execution sequencing

- Propose day-by-day order with parallel work windows.
- Ensure high-risk discovery tasks happen before implementation tasks.

Output:
- Ordered weekly run plan

### 6) Weekly status and KPI report

- Generate status sections:
  - Delivered this period
  - In progress
  - Blockers and risks
  - Decisions needed
  - Next actions
- Include mandatory KPIs:
  - planned vs completed
  - spillover count
  - blocker count
  - risk count
  - test trend
  - coverage trend
  - lint/typecheck trend

Output:
- Weekly status update

### 7) Reprioritization and next-week handoff

- Convert unresolved items into next-week backlog candidates.
- Mark each as must-do, should-do, or can-wait.
- Record assumptions and open decisions.

Output:
- Next-week backlog draft and decision log

## Decision Points and Branching Logic

1. If roadmap target is missing:
- stop sprint decomposition
- ask for target week or phase
- provide a minimal fallback plan draft flagged as assumption-based

2. If reports evidence is missing:
- mark related story status as missing evidence
- do not claim completion
- create a corrective task to generate missing report artifacts

3. If scope change appears mid-week:
- classify as minor, moderate, major
- minor: adjust subtasks only
- moderate: re-estimate stories and re-sequence
- major: freeze non-critical scope and request stakeholder decision

4. If dependency is blocked:
- move dependent story to blocked
- schedule alternative unblocked story to protect throughput
- emit escalation note with owner and due date

5. If quality trend degrades (tests, coverage, lint/typecheck):
- insert quality recovery story in next sprint
- reduce new scope until baseline stabilizes

## Completion Checks

A sprint plan is complete only if all checks pass:

- roadmap phase and week are explicitly stated
- every story includes all mandatory Jira fields
- every deliverable has an evidence file target in reports/
- risks have mitigation and escalation triggers
- execution order is explicit
- status report includes all mandatory KPIs
- open assumptions and decisions are listed

## Output Templates

### A) Sprint Plan Template

- Sprint Goal
- Success Criteria
- Epic
- Story List (full mandatory fields)
- Dependencies
- Risks and mitigations
- Execution order by day
- Sprint Definition of Done

### B) Weekly Status Template

- Delivered
- In progress
- Blockers and risks
- Decisions needed
- Next actions
- KPI block (delivery + quality)

## Anti-Patterns to Avoid

- claiming completion without evidence in reports/
- stories without acceptance criteria
- generic plans without owners or estimates
- ignoring dependencies during sequencing
- reporting status without KPIs

## References

- Asana PM skills article: https://asana.com/pt/resources/project-management-skills
- Project roadmap and planning files in roadmap/
- Execution evidence files in reports/
