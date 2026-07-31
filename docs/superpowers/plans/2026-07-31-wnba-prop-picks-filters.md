# WNBA Prop Picks Filters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Client-side multi-select filters (Stat, O/U, Team) on Prop Picks plus disabled +EV Soon control.

**Architecture:** Pure `filterPropLines` helper; `PropPicksFilters` toolbar with multi-check dropdowns; page wires filtered rows into `PropPicksTable` with distinct empty-filter copy.

**Tech Stack:** React, Vitest, Testing Library

## Global Constraints

- Multi-select; empty = all for that dimension
- AND across filters; OR within
- +EV disabled: `+EV · Soon`
- Client-side only; no API/URL sync
- Filtered empty ≠ API unavailable

## File structure

| File | Responsibility |
| --- | --- |
| `filterPropLines.ts` | Pure filter + option extractors |
| `PropPicksFilters.tsx` | Toolbar UI |
| `PropPicksTable.tsx` | `emptyMessage` / filtered empty |
| `LeaguePropPicksPage.tsx` | State + wire |

---

### Task 1: filterPropLines helper

- [ ] Write failing tests
- [ ] Implement
- [ ] Pass + commit

### Task 2: Filters UI + page wiring

- [ ] Failing tests for filters + table empty states
- [ ] Implement PropPicksFilters + page + table message prop
- [ ] Pass + commit
