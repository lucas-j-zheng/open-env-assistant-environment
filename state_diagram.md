# Personal Assistant Calendar Environment — State Diagram

```mermaid
stateDiagram-v2
    direction TB

    [*] --> Reset: reset(seed)

    state Reset {
        direction LR
        r1: Clear all state
        r2: Pick episode date<br/>(weekday from seed)
        r3: generate_episode_config()<br/>seed events + inbox + personal
        r4: Mark initial conflicts
        r1 --> r2 --> r3 --> r4
    }

    Reset --> StepLoop: Markov Observation<br/>"Good morning. Check inbox + calendar."

    state StepLoop {
        direction TB

        Receive: Receive Action<br/>(JSON tool call)

        state Interrupts <<choice>>
        Receive --> Interrupts: step_count++

        state "Process Interrupts" as PI {
            direction LR
            i_ceo: new_meeting<br/>CEO Sync at 15:00
            i_inbox: inbox_update<br/>Contradicting message
            i_cancel: cancellation<br/>Lunch meeting cancelled
            i_avail: availability_change<br/>Bob Wed afternoons blocked
            i_resched: reschedule_request<br/>Move morning meeting to 11:00
            i_personal: personal_event_change<br/>Partner texts: time changed
            i_policy: policy_change<br/>Description required (>30min)
        }

        Interrupts --> PI: step matches interrupt
        Interrupts --> Parse: no interrupt

        PI --> Parse

        state Parse <<choice>>
        Parse --> Dispatch: valid JSON tool call
        Parse --> ErrorMsg: not valid JSON

        state "Dispatch Tool" as Dispatch {
            direction LR
            t_read: READ tools<br/>list_events<br/>find_free_slots<br/>check_conflicts<br/>check_availability<br/>get_task_list<br/>get_constraints<br/>check_constraint_violations<br/>read_inbox<br/>check_personal_calendar
            t_write: WRITE tools<br/>create_event (+ negotiation gate)<br/>delete_event (immovable check)<br/>edit_event (immovable check)<br/>resolve_conflict (immovable check)<br/>send_notification<br/>reply_message (keyword validation)
            t_discover: DISCOVERY tools<br/>get_contact_preferences<br/>(tracks discovered prefs)
        }

        ErrorMsg --> BuildObs

        Dispatch --> CheckCompletions

        state "Check Completions (18 flags)" as CheckCompletions {
            direction TB

            state "Static Flags" as Static {
                direction LR
                f1: standup_scheduled<br/>Alice+Bob, ≤30min<br/>+ negotiation resolved
                f2: focus_time_booked<br/>today, ≥60min, no overlaps
                f4: reminder_set<br/>dentist next week, afternoon
                f5: meeting_cancelled<br/>deleted + notified attendee
                f9: kickoff_scheduled<br/>Alice+Bob+Eve<br/>+ negotiation resolved
                f_inbox: inbox_cleared<br/>all visible messages replied
                f_diplo: diplomatic_reply_sent<br/>tough email answered
                f_wl: work_life_conflicts_resolved<br/>no personal/work overlaps
            }

            state "Dynamic Flags (from interrupts)" as Dynamic {
                direction LR
                f6: ceo_sync_accommodated<br/>CEO at 15:00, no conflicts
                f7: cancellation_handled<br/>lunch meeting deleted
                f8: reschedule_handled<br/>morning meeting at 11:00
                f_avail: availability_drift_handled<br/>Bob Wed re-validated
                f_client: client_request_updated<br/>contradiction replied to
                f_pers: personal_update_handled<br/>no overlaps after change
                f_desc: description_policy_met<br/>all >30min have agenda
            }

            state "Revocable Flags (can be lost)" as Revocable {
                direction LR
                f3: conflicts_resolved<br/>(lost if new overlap)
                f10: hard_constraints_clear<br/>(lost if new violation)
                f11: preferences_optimized<br/>(lost if preferences break)
                note1: Also revocable:<br/>focus_time_booked<br/>reminder_set<br/>meeting_cancelled<br/>ceo_sync_accommodated<br/>cancellation_handled<br/>reschedule_handled<br/>availability_drift_handled<br/>personal_update_handled<br/>description_policy_met<br/>inbox_cleared<br/>work_life_conflicts_resolved
            }
        }

        CheckCompletions --> RefreshInterrupts

        state "Refresh Unhandled Interrupts" as RefreshInterrupts {
            direction LR
            ri: Remove interrupts whose<br/>flags are now earned
        }

        RefreshInterrupts --> BuildObs

        state "Build Markov Observation" as BuildObs {
            direction LR
            bo: output + interrupt msg<br/>+ state_summary text<br/>+ calendar_snapshot<br/>+ discovered_preferences<br/>+ constraint_status<br/>+ negotiations + inbox<br/>reward = flags / 18<br/>done = (flags == 18)
        }
    }

    BuildObs --> Done: done == true
    BuildObs --> Receive: done == false

    Done --> [*]

    note right of Reset
        Seed generator produces per-episode:
        - Work events (with intentional conflict)
        - Personal events (immovable)
        - Inbox messages (scheduling + diplomatic)
        - Randomized interrupt steps
        - Dynamic meeting titles
    end note

    note right of CheckCompletions
        Hard constraints:
        - Bob cannot attend on Mondays
        - No meetings during 12:00-13:00 lunch
        - Eve unavailable before 10:00
        - Meetings >30min need description (when active)

        Soft constraints:
        - Alice prefers mornings (<12:00)
        - Charlie prefers afternoons (>13:00)
        - Max 3 meetings per person per day

        Visibility:
        - Public: lunch, max-3, description (when active)
        - Private: Bob/Eve/Alice/Charlie (via get_contact_preferences)
    end note
```

## Reward Function

```
reward = |flags_found| / 18      (range: 0.0 to 1.0)
done   = (|flags_found| == 18)
```

## Step Timeline with Interrupts

Interrupt steps are randomized per seed. Default ordering:

```
Step 1  ─── agent acts ───────────────────────────
Step 2  ─── agent acts ───────────────────────────
Step 3  ─── INTERRUPT: CEO Sync added at 15:00 ──  (new conflict)
Step 4  ─── agent acts ───────────────────────────
Step 5  ─── INTERRUPT: Contradicting inbox msg ──  (follow-up changes earlier request)
Step 6  ─── INTERRUPT: Lunch meeting cancelled ──  (slot freed)
Step 7  ─── INTERRUPT: Bob Wed unavailable ──────  (schedule drift)
Step 8  ─── agent acts ───────────────────────────
Step 9  ─── INTERRUPT: Move morning mtg to 11:00 ─
Step 10 ─── INTERRUPT: Personal event time change  (partner texts)
Step 11 ─── agent acts ───────────────────────────
Step 12 ─── INTERRUPT: Description policy active ─ (HR policy)
  ...
Step 60 ─── max steps (agent loop limit) ─────────
```

## Flag Dependencies

```
No prerequisites             Requires interrupt fired           Revocable (state-dependent)
─────────────────            ──────────────────────────         ──────────────────────────
standup_scheduled             ceo_sync_accommodated              conflicts_resolved
focus_time_booked             cancellation_handled               hard_constraints_clear
reminder_set                  reschedule_handled                 preferences_optimized
meeting_cancelled             availability_drift_handled         focus_time_booked
kickoff_scheduled             client_request_updated             reminder_set
inbox_cleared                 personal_update_handled            meeting_cancelled
diplomatic_reply_sent         description_policy_met             ceo_sync_accommodated
work_life_conflicts_resolved                                     cancellation_handled
                                                                 reschedule_handled
                                                                 availability_drift_handled
                                                                 personal_update_handled
                                                                 description_policy_met
                                                                 inbox_cleared
                                                                 work_life_conflicts_resolved
```

## Observation Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   CalendarObservation                    │
├─────────────────────────────────────────────────────────┤
│ Core:                                                   │
│   output          ← tool result + interrupt text        │
│   reward, done    ← computed from flags                 │
│   flags_found     ← earned completion flags             │
├─────────────────────────────────────────────────────────┤
│ Markov State (makes observation self-contained):        │
│   state_summary        ← ~500-1200 token text render    │
│   calendar_snapshot    ← all events (structured)        │
│   discovered_preferences ← only queried people          │
│   discovered_constraints ← private constraints found    │
│   constraint_status    ← hard/soft violation report     │
│   active_negotiations  ← round, attempts, feedback      │
│   resolved_negotiations ← success/failure per scenario  │
│   unhandled_interrupts ← auto-cleared when resolved     │
│   notifications_sent   ← full notification log          │
│   inbox_snapshot       ← messages visible at step       │
│   step_count           ← current step number            │
└─────────────────────────────────────────────────────────┘

Agent sliding window pattern:
  messages = [system_prompt, state_summary] + history[-6:]
  (~300 + ~800 + ~1200 = ~2300 tokens per turn)
```

## Negotiation Flow

```
create_event("standup", Alice+Bob)
    │
    ├── 1st attempt → ALWAYS rejected (round 0 message)
    │                  "Bob: keep it under 20 minutes"
    │
    ├── 2nd attempt (duration > 20) → rejected again + hint
    │                  "check get_contact_preferences for Alice"
    │
    ├── 2nd attempt (duration ≤ 20) → ACCEPTED
    │   └── Event created, negotiation resolved True
    │
    └── 4th attempt → FAILED (max_rounds exceeded)
        └── Event NOT created, negotiation resolved False

create_event("kickoff", Alice+Bob+Eve)
    │
    ├── 1st attempt → ALWAYS rejected (round 0)
    │                  "Eve: 45min max + 30min buffer"
    │
    ├── 2nd attempt (≤45min + buffer OK) → round 0 passes
    │   ├── start_time ≥ 12:00 → round 1 triggered
    │   │   └── "Alice: move to morning"
    │   └── start_time < 12:00 → round 1 skipped → ACCEPTED
    │
    └── 5th attempt → FAILED (max_rounds exceeded)
```
