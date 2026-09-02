import argparse
import csv
import os
import sys
from ortools.sat.python import cp_model

# --- Configuration -------------------------------------------------------
# Chores that may be assigned to MULTIPLE people in the same week.
# Any chore not listed here is assigned to exactly one person.
CHORE_ASSIGNMENT_COUNTS = {
    "Registration 8-9": 2,
    "Registration 9-10": 2,
}
DEFAULT_ASSIGNMENT_COUNT = 1

# Pairs of chores a person cannot hold in the same week, nor in adjacent
# weeks (in either direction — e.g. Reg 9-10 this week, Reg 8-9 next week).
CONFLICT_PAIRS = [
    (
        "Registration 8-9",
        "Registration 9-10",
    ),
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Assign weekly chores fairly based on preferences."
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file containing names and chore preferences.",
    )
    parser.add_argument(
        "output_csv", help="Path where the output schedule CSV should be saved."
    )
    parser.add_argument(
        "-w",
        "--weeks",
        type=int,
        default=4,
        help="Number of weeks to generate assignments for (default: 4).",
    )
    parser.add_argument(
        "--runtime",
        type=int,
        default=60,
        help="Seconds to allow solver to run for (default: 60).",
    )
    return parser.parse_args()


def load_preferences(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Input file '{file_path}' does not exist.")
        sys.exit(1)

    people = []
    chores = []
    preferences = {}

    with open(file_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        chores = header[1:]

        for row in reader:
            if not row:
                continue
            person = row[0].strip()
            people.append(person)
            for i, response in enumerate(row[1:]):
                chore = chores[i]
                preferences[(person, chore)] = response.strip().lower() == "yes"

    return people, chores, preferences


def optimize_schedule(people, chores, preferences, total_weeks, runtime):
    model = cp_model.CpModel()

    # 1. Variables: x[w, p, c] == 1 if person p does chore c in week w
    x = {}
    for w in range(total_weeks):
        for p in people:
            for c in chores:
                x[w, p, c] = model.NewBoolVar(f"x_{w}_{p}_{c}")

                # Rule: Respect "No" choices absolutely
                if not preferences[(p, c)]:
                    model.Add(x[w, p, c] == 0)

    # 2. Constraint: assign each chore to its configured number of people
    for w in range(total_weeks):
        for c in chores:
            target = CHORE_ASSIGNMENT_COUNTS.get(c, DEFAULT_ASSIGNMENT_COUNT)
            willing = [p for p in people if preferences[(p, c)]]
            if len(willing) < target:
                print(
                    f"Error: chore '{c}' needs {target} assignees per week, "
                    f"but only {len(willing)} people are willing."
                )
                sys.exit(1)
            model.Add(sum(x[w, p, c] for p in people) == target)

    # 3. Constraint: Every person must have between 1 and 2 chores per week
    for w in range(total_weeks):
        for p in people:
            weekly_chore_count = sum(x[w, p, c] for c in chores)
            model.Add(weekly_chore_count >= 1)
            model.Add(weekly_chore_count <= 2)

    # 4. New Rule: Minimize consecutive-week repeats of the exact same chore
    # Soft constraint tracked via a penalty variable
    consecutive_repeats = []
    for w in range(total_weeks - 1):
        for p in people:
            for c in chores:
                is_consecutive = model.NewBoolVar(f"consec_{w}_{p}_{c}")
                # is_consecutive can only be true if assigned in week w AND week w+1
                model.Add(x[w, p, c] + x[w + 1, p, c] <= 1).OnlyEnforceIf(
                    is_consecutive.Not()
                )
                model.Add(x[w, p, c] + x[w + 1, p, c] == 2).OnlyEnforceIf(
                    is_consecutive
                )
                consecutive_repeats.append(is_consecutive)

    # 4b. Conflict pairs: not in the same week, not in adjacent weeks
    for chore_a, chore_b in CONFLICT_PAIRS:
        unknown = [c for c in (chore_a, chore_b) if c not in chores]
        if unknown:
            print(
                f"Error: conflict pair references unknown chore(s): {unknown}. "
                "Check spelling against the CSV header."
            )
            sys.exit(1)

        for w in range(total_weeks):
            for p in people:
                # Never both in the same week
                model.Add(x[w, p, chore_a] + x[w, p, chore_b] <= 1)

                # Never chore_a this week and chore_b next week (either order)
                if w + 1 < total_weeks:
                    model.Add(x[w, p, chore_a] + x[w + 1, p, chore_b] <= 1)
                    model.Add(x[w, p, chore_b] + x[w + 1, p, chore_a] <= 1)

    # 5. Tracking total repeats of any chore over the entire schedule window
    times_chore_done = {}
    for p in people:
        for c in chores:
            times_chore_done[p, c] = model.NewIntVar(0, total_weeks, f"count_{p}_{c}")
            model.Add(
                times_chore_done[p, c] == sum(x[w, p, c] for w in range(total_weeks))
            )

    # 5b. Chore groups: paired chores count as ONE chore for repeat tracking
    groups = {c: [c] for c in chores}  # start as singletons
    for chore_a, chore_b in CONFLICT_PAIRS:
        # Both members fold into one group keyed by the pair itself
        groups[(chore_a, chore_b)] = [chore_a, chore_b]
        del groups[chore_a]
        del groups[chore_b]

    # Weekly group membership: y[w, p, g] == 1 if person p does ANY chore
    # from group g in week w. (Safe: the conflict constraint already caps
    # paired groups at one member per week, but AddMaxEquality works either way.)
    y = {}
    for p in people:
        for g_name, members in groups.items():
            for w in range(total_weeks):
                y[w, p, g_name] = model.NewBoolVar(f"grp_{w}_{p}_{members[0][:10]}")
                model.AddMaxEquality(y[w, p, g_name], [x[w, p, c] for c in members])

    # Penalize ANY repeat of the same (person, group) pair across the schedule
    repeat_pairs = []
    for p in people:
        for g_name in groups:
            for w1 in range(total_weeks):
                for w2 in range(w1 + 1, total_weeks):
                    z = model.NewBoolVar(
                        f"rep_{w1}_{w2}_{p}_{g_name[:12]}_{g_name[-12:]}"
                    )
                    model.AddMultiplicationEquality(
                        z, [y[w1, p, g_name], y[w2, p, g_name]]
                    )
                    repeat_pairs.append(z)

    # Track the absolute maximum number of times any single person repeats any chore
    max_individual_chore_repeat = model.NewIntVar(0, total_weeks, "max_ind_repeat")
    for p in people:
        for c in chores:
            model.Add(max_individual_chore_repeat >= times_chore_done[p, c])

    # 6. Tracking overall workload fairness (total tasks across all weeks)
    total_tasks_per_person = {
        p: model.NewIntVar(0, total_weeks * len(chores), f"total_{p}") for p in people
    }
    for p in people:
        model.Add(
            total_tasks_per_person[p]
            == sum(x[w, p, c] for w in range(total_weeks) for c in chores)
        )

    min_total = model.NewIntVar(0, total_weeks * len(chores), "min_total")
    max_total = model.NewIntVar(0, total_weeks * len(chores), "max_total")
    for p in people:
        model.Add(min_total <= total_tasks_per_person[p])
        model.Add(max_total >= total_tasks_per_person[p])

    # 7. NEW OBJECTIVE PRIORITY:
    # Heavy Weight (10,000): Keep the maximum cap of any single chore repetition as low as mathematically possible.
    # Medium Weight (100): Eliminate back-to-back consecutive repeats.
    # Light Weight (10): Minimize the difference between total overall task workloads.
    model.Minimize(
        10000 * max_individual_chore_repeat
        + 1000 * sum(consecutive_repeats)
        + 1000 * sum(repeat_pairs)
        + 10 * (max_total - min_total)
    )

    # Solve the model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = runtime
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        schedule = {w: {p: [] for p in people} for w in range(total_weeks)}
        for w in range(total_weeks):
            for p in people:
                for c in chores:
                    if solver.Value(x[w, p, c]) == 1:
                        schedule[w][p].append(c)
        return schedule
    else:
        print(
            "Error: Could not find a valid schedule matching these repetition constraints."
        )
        sys.exit(1)


def save_schedule(schedule, people, total_weeks, output_path):
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Week"] + people)

        for w in range(total_weeks):
            row = [f"Week {w + 1}"]
            for p in people:
                chores_assigned = schedule[w][p]
                row.append("; ".join(chores_assigned) if chores_assigned else "None")
            writer.writerow(row)

    print(f"Success! Schedule successfully saved to: {output_path}")


def main():
    args = parse_arguments()
    people, chores, preferences = load_preferences(args.input_csv)

    print(f"Loaded {len(people)} people and {len(chores)} chores.")
    print(f"Generating optimized schedule with heavy anti-repetition rules...")

    schedule = optimize_schedule(people, chores, preferences, args.weeks, args.runtime)
    save_schedule(schedule, people, args.weeks, args.output_csv)


if __name__ == "__main__":
    main()
