# Personal Assistant Calendar Environment — State Diagram

```mermaid
stateDiagram-v2
    direction TB

    [*] --> Reset: reset(seed)

    state Reset {
        direction LR
        r1: Clear state
        r2: Pick episode date (weekday from seed)
        r3: Seed 4 events
        r4: Mark initial conflicts
        r1 --> r2 --> r3 --> r4
    }

    Reset --> StepLoop: Initial Observation<br/>"4 events, 1 conflict"

    state StepLoop {
        direction TB

        Receive: Receive Action<br/>(JSON tool call)

        state Interrupts <<choice>>
        Receive --> Interrupts: step_count++

        state "Process Interrupts" as PI {
            direction LR
            i3: Step 3<br/>CEO Sync injected at 15:00<br/>(new conflict)
            i6: Step 6<br/>Lunch with Client cancelled<br/>(slot freed)
            i9: Step 9<br/>Alice requests Morning Sync<br/>move to 11:00
        }

        Interrupts --> PI: step matches 3, 6, or 9
        Interrupts --> Parse: no interrupt

        PI --> Parse

        state Parse <<choice>>
        Parse --> Dispatch: valid JSON tool call
        Parse --> ErrorMsg: not valid JSON

        state "Dispatch Tool" as Dispatch {
            direction LR
            t_read: READ tools<br/>list_events<br/>find_free_slots<br/>check_conflicts<br/>check_availability<br/>get_task_list<br/>get_constraints<br/>check_constraint_violations
            t_write: WRITE tools<br/>create_event<br/>delete_event<br/>edit_event<br/>resolve_conflict<br/>send_notification
        }

        ErrorMsg --> BuildObs

        Dispatch --> CheckCompletions

        state "Check Completions (11 flags)" as CheckCompletions {
            direction TB

            state "Static Flags (earned once)" as Static {
                direction LR
                f1: standup_scheduled<br/>Alice+Bob, ≤45min<br/>no conflicts
                f2: focus_time_booked<br/>today, ≥60min<br/>no overlaps
                f4: reminder_set<br/>dentist next week<br/>afternoon
                f5: meeting_cancelled<br/>deleted + notified attendee
                f9: kickoff_scheduled<br/>Alice+Bob+Eve<br/>no hard violations
            }

            state "Dynamic Flags (from interrupts)" as Dynamic {
                direction LR
                f6: ceo_sync_accommodated<br/>CEO Sync at 15:00<br/>no conflicts
                f7: cancellation_handled<br/>Lunch with Client deleted
                f8: reschedule_handled<br/>Morning Sync at 11:00
            }

            state "Revocable Flags (can be lost)" as Revocable {
                direction LR
                f3: conflicts_resolved<br/>no overlaps today<br/>(lost if new overlap)
                f10: hard_constraints_clear<br/>0 hard violations<br/>(lost if new violation)
                f11: preferences_optimized<br/>≥2 soft satisfied<br/>(lost if preferences break)
            }
        }

        CheckCompletions --> BuildObs

        state "Build Observation" as BuildObs {
            direction LR
            bo: output + interrupt msg<br/>reward = flags / 11<br/>done = (flags == 11)
        }
    }

    BuildObs --> Done: done == true
    BuildObs --> Receive: done == false

    Done --> [*]

    note right of Reset
        Seed events (all on "today"):
        1. Morning Sync 10:00-10:30 [Alice, Charlie]
        2. Lunch with Client 12:00-13:00 [Dave]
        3. Old Project Review 14:00-15:00 [Alice, Bob, Charlie]
        4. Design Review 14:30-15:30 [Eve]
        --- CONFLICT: #3 overlaps #4 ---
    end note

    note right of CheckCompletions
        Hard constraints:
        - Bob cannot attend on Mondays
        - No meetings during 12:00-13:00 lunch
        - Eve unavailable before 10:00

        Soft constraints:
        - Alice prefers mornings (<12:00)
        - Charlie prefers afternoons (>13:00)
        - Max 3 meetings per person per day
    end note
```

## Reward Function

```
reward = |flags_found| / 11      (range: 0.0 to 1.0)
done   = (|flags_found| == 11)
```

## Step Timeline with Interrupts

```
Step 1  ─── agent acts ───────────────────────────
Step 2  ─── agent acts ───────────────────────────
Step 3  ─── INTERRUPT: CEO Sync added at 15:00 ──  (new conflict with Design Review)
Step 4  ─── agent acts ───────────────────────────
Step 5  ─── agent acts ───────────────────────────
Step 6  ─── INTERRUPT: Lunch with Client cancelled  (12:00-13:00 freed)
Step 7  ─── agent acts ───────────────────────────
Step 8  ─── agent acts ───────────────────────────
Step 9  ─── INTERRUPT: Move Morning Sync to 11:00 ─
Step 10 ─── agent acts ───────────────────────────
  ...
Step 30 ─── max steps (agent loop limit) ─────────
```

## Flag Dependencies

```
No prerequisites           Requires interrupt fired     Revocable (state-dependent)
─────────────────          ──────────────────────────   ──────────────────────────
standup_scheduled          ceo_sync_accommodated (s3)   conflicts_resolved
focus_time_booked          cancellation_handled  (s6)   hard_constraints_clear
reminder_set               reschedule_handled    (s9)   preferences_optimized
meeting_cancelled
kickoff_scheduled
```
