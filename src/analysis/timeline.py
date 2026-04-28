"""
Agent Task Timeline Generator

Generates an interactive HTML timeline showing which agent executes which task over time.
Horizontal axis = time, vertical axis = agents, colored blocks = tasks.

Usage:
    python analysis/timeline.py <log_csv_path> [--max-cases N] [--output file.html]

Examples:
    python analysis/timeline.py results/remote/mappo_ic3net/20260209_003518/logs/log_20260209_003526.csv
    python analysis/timeline.py results/remote/mappo_ic3net/20260209_003518/logs/log_20260209_003526.csv --max-cases 10
"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path


def parse_timestamp(ts_str):
    """Parse ISO timestamp string to epoch seconds."""
    ts_str = ts_str.strip()
    # Remove nanosecond precision (Python datetime can't handle it)
    # Format: 2019-03-25 08:00:01.123456789+00:00
    if '.' in ts_str and '+' in ts_str:
        dot_pos = ts_str.index('.')
        plus_pos = ts_str.index('+', dot_pos)
        # Keep only 6 decimal places
        decimals = ts_str[dot_pos+1:plus_pos][:6]
        ts_str = ts_str[:dot_pos+1] + decimals + ts_str[plus_pos:]
    elif '.' in ts_str and '-' in ts_str[ts_str.index('.'):]:
        dot_pos = ts_str.index('.')
        # Find the timezone minus (not date minus)
        rest = ts_str[dot_pos+1:]
        minus_pos = rest.index('-')
        decimals = rest[:minus_pos][:6]
        ts_str = ts_str[:dot_pos+1] + decimals + '-' + rest[minus_pos+1:]

    for fmt in [
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]:
        try:
            return datetime.strptime(ts_str, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts_str}")


def load_log(csv_path, max_cases=None):
    """Load a log CSV and return structured task data."""
    tasks = []
    agents_set = set()
    case_ids = set()

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = int(row['case_id'])
            if max_cases is not None and case_id >= max_cases:
                continue

            case_ids.add(case_id)
            task_name = row['task_name']
            task_id = int(row['task_id'])
            agents_required = int(row['task_agents_required'])

            # Parse agent IDs and names (comma-separated for collab tasks)
            agent_ids = [x.strip() for x in row['task_agent_id'].split(',')]
            agent_names = [x.strip() for x in row['task_agent_name'].split(',')]

            start_ts = parse_timestamp(row['task_started_time'])
            end_ts = parse_timestamp(row['task_completed_time'])
            assigned_ts = parse_timestamp(row['task_assigned_time'])

            # Skip zero-duration artificial tasks
            if abs(end_ts - start_ts) < 0.5:
                continue

            # Skip artificial/system agents
            for aid, aname in zip(agent_ids, agent_names):
                if 'artificial' in aname or 'Pharmacy System' in aname:
                    continue

                agents_set.add(aname)
                tasks.append({
                    'case_id': case_id,
                    'task_id': task_id,
                    'task_name': task_name,
                    'agent_name': aname,
                    'start': start_ts,
                    'end': end_ts,
                    'assigned': assigned_ts,
                    'waiting_time': max(0, start_ts - assigned_ts),
                    'is_collab': agents_required > 1,
                    'agents_required': agents_required,
                    'all_agents': ', '.join(n for n in agent_names if 'artificial' not in n and 'Pharmacy System' not in n),
                })

    return tasks, sorted(agents_set), sorted(case_ids)


# Short task name mapping
TASK_SHORT = {
    'Process drop-off': 'Drop-off',
    'Enter prescription details': 'Enter Rx',
    'Pack the drugs (Production)': 'Pack',
    'Check for Quality Assurance': 'QA Check',
    'Pick-up': 'Pick-up',
    'Resolve DUR manually': 'DUR',
    'Check if refill is allowed': 'Refill?',
    'Check DUR': 'DUR Check',
    'Check Insurance': 'Insurance',
    'Prescription received': 'Rx Recv',
    'Prescription fulfilled': 'Rx Done',
}


def generate_html(tasks, agents, case_ids, title="Agent Task Timeline"):
    """Generate a standalone interactive HTML timeline."""

    # Compute global time range
    if not tasks:
        return "<html><body>No tasks found.</body></html>"

    min_time = min(t['start'] for t in tasks)
    max_time = max(t['end'] for t in tasks)
    total_duration = max_time - min_time

    # Assign consistent colors per case
    n_cases = len(case_ids)

    # Prepare task data for JavaScript
    js_tasks = []
    for t in tasks:
        short_name = TASK_SHORT.get(t['task_name'], t['task_name'][:12])
        label = f"C{t['case_id']}.T{t['task_id']}"

        js_tasks.append({
            'caseId': t['case_id'],
            'taskId': t['task_id'],
            'taskName': t['task_name'],
            'shortName': short_name,
            'label': label,
            'agent': t['agent_name'],
            'start': t['start'] - min_time,  # Relative seconds
            'end': t['end'] - min_time,
            'waiting': t['waiting_time'],
            'isCollab': t['is_collab'],
            'allAgents': t['all_agents'],
        })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8f9fa; color: #333; }}

.header {{
    position: sticky; top: 0; z-index: 100;
    background: #fff; border-bottom: 1px solid #ddd; padding: 12px 20px;
    display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
}}
.header h1 {{ font-size: 18px; font-weight: 600; }}
.header .stats {{ font-size: 13px; color: #666; }}

.controls {{
    display: flex; align-items: center; gap: 12px; font-size: 13px;
}}
.controls label {{ color: #555; }}
.controls input[type=range] {{ width: 140px; }}
.controls button {{
    padding: 4px 12px; border: 1px solid #ccc; background: #fff;
    border-radius: 4px; cursor: pointer; font-size: 12px;
}}
.controls button:hover {{ background: #f0f0f0; }}

.timeline-wrapper {{
    overflow-x: scroll; overflow-y: hidden;
    position: relative; margin-top: 0;
}}

.timeline-container {{
    position: relative;
    min-height: calc(100vh - 80px);
}}

.agent-labels {{
    position: sticky; left: 0; z-index: 50;
    width: 160px; float: left;
    background: #fff; border-right: 2px solid #ddd;
}}
.agent-label {{
    height: 56px; display: flex; align-items: center;
    padding: 0 12px; font-size: 12px; font-weight: 600;
    border-bottom: 1px solid #eee; color: #444;
}}

.timeline-area {{
    margin-left: 160px; position: relative;
}}

.agent-row {{
    height: 56px; position: relative;
    border-bottom: 1px solid #eee;
}}
.agent-row:nth-child(even) {{ background: rgba(0,0,0,0.015); }}

.time-axis {{
    height: 28px; position: sticky; top: 56px; z-index: 40;
    background: #fafafa; border-bottom: 1px solid #ddd;
    margin-left: 160px;
}}
.time-tick {{
    position: absolute; top: 0; height: 100%;
    font-size: 10px; color: #888; padding-top: 4px; padding-left: 4px;
    border-left: 1px solid #ddd;
}}

.task-block {{
    position: absolute; height: 38px; top: 9px;
    border-radius: 4px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 600; color: #fff;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
    transition: opacity 0.15s, box-shadow 0.15s;
    border: 1px solid rgba(0,0,0,0.15);
    min-width: 2px;
}}
.task-block:hover {{
    opacity: 0.85 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    z-index: 30 !important;
}}
.task-block.collab {{
    border: 2px solid #333;
    border-style: dashed;
}}
.task-block.dimmed {{ opacity: 0.15 !important; }}

.waiting-block {{
    position: absolute; height: 38px; top: 9px;
    background: repeating-linear-gradient(45deg, transparent, transparent 3px, rgba(0,0,0,0.06) 3px, rgba(0,0,0,0.06) 6px);
    border-radius: 4px 0 0 4px;
    border: 1px dashed rgba(0,0,0,0.15);
    pointer-events: none;
}}

.tooltip {{
    display: none; position: fixed; z-index: 200;
    background: #333; color: #fff; padding: 10px 14px;
    border-radius: 6px; font-size: 12px; line-height: 1.6;
    pointer-events: none; max-width: 320px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}}
.tooltip.visible {{ display: block; }}
.tooltip .label {{ font-weight: 700; font-size: 13px; margin-bottom: 4px; }}
.tooltip .detail {{ color: #ccc; }}

.legend {{
    position: sticky; left: 0;
    display: flex; gap: 12px; padding: 8px 20px; flex-wrap: wrap;
    background: #fff; border-top: 1px solid #ddd; font-size: 11px;
}}
.legend-item {{
    display: flex; align-items: center; gap: 4px; cursor: pointer;
}}
.legend-item .swatch {{
    width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.15);
}}
.legend-item.dimmed .swatch {{ opacity: 0.2; }}
.legend-item.dimmed {{ color: #aaa; }}

.playhead {{
    position: absolute; top: 0; width: 2px;
    background: red; z-index: 45; pointer-events: none;
    display: none;
}}
</style>
</head>
<body>

<div class="header">
    <h1>{title}</h1>
    <div class="stats">{len(tasks)} tasks &middot; {n_cases} cases &middot; {len(agents)} agents &middot; {total_duration/60:.0f} min</div>
    <div class="controls">
        <label>Zoom:</label>
        <input type="range" id="zoomSlider" min="0.5" max="20" step="0.1" value="3">
        <span id="zoomLabel">3.0 px/s</span>
        <button onclick="resetZoom()">Reset</button>
        <button onclick="fitAll()">Fit All</button>
        <label style="margin-left:12px;">Show waiting:</label>
        <input type="checkbox" id="showWaiting" checked>
    </div>
</div>

<div class="legend" id="legend"></div>

<div class="time-axis" id="timeAxis"></div>

<div class="timeline-wrapper" id="wrapper">
    <div class="timeline-container" id="container">
        <div class="agent-labels" id="agentLabels"></div>
        <div class="timeline-area" id="timelineArea"></div>
    </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const TASKS = {json.dumps(js_tasks)};
const AGENTS = {json.dumps(agents)};
const TOTAL_DURATION = {total_duration};
const N_CASES = {n_cases};

// Case color palette (perceptually distinct)
const PALETTE = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#e6beff',
    '#1abc9c', '#e74c3c', '#3498db', '#9b59b6', '#2ecc71',
    '#e67e22', '#1f618d', '#c0392b', '#27ae60', '#8e44ad',
];

function caseColor(caseId) {{
    return PALETTE[caseId % PALETTE.length];
}}

let pxPerSecond = 3;
let activeCases = new Set();
TASKS.forEach(t => activeCases.add(t.caseId));

// Build agent labels
const labelsEl = document.getElementById('agentLabels');
AGENTS.forEach(agent => {{
    const d = document.createElement('div');
    d.className = 'agent-label';
    d.textContent = agent;
    labelsEl.appendChild(d);
}});

// Legend
const legendEl = document.getElementById('legend');
const caseIds = [...new Set(TASKS.map(t => t.caseId))].sort((a,b) => a - b);
caseIds.forEach(cid => {{
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<div class="swatch" style="background:${{caseColor(cid)}}"></div>Case ${{cid}}`;
    item.onclick = () => {{
        if (activeCases.has(cid)) activeCases.delete(cid);
        else activeCases.add(cid);
        item.classList.toggle('dimmed', !activeCases.has(cid));
        renderTasks();
    }};
    legendEl.appendChild(item);
}});

function renderTimeAxis() {{
    const axisEl = document.getElementById('timeAxis');
    axisEl.innerHTML = '';
    const totalWidth = TOTAL_DURATION * pxPerSecond;
    axisEl.style.width = totalWidth + 'px';

    // Determine tick interval
    let interval;
    if (pxPerSecond > 5) interval = 60;        // 1 min
    else if (pxPerSecond > 1) interval = 300;   // 5 min
    else if (pxPerSecond > 0.3) interval = 600; // 10 min
    else interval = 1800;                        // 30 min

    for (let s = 0; s <= TOTAL_DURATION; s += interval) {{
        const tick = document.createElement('div');
        tick.className = 'time-tick';
        tick.style.left = (s * pxPerSecond) + 'px';
        const mins = Math.floor(s / 60);
        const hrs = Math.floor(mins / 60);
        const m = mins % 60;
        tick.textContent = hrs > 0 ? `${{hrs}}h${{String(m).padStart(2,'0')}}m` : `${{mins}}m`;
        axisEl.appendChild(tick);
    }}
}}

function renderTasks() {{
    const area = document.getElementById('timelineArea');
    area.innerHTML = '';
    const totalWidth = TOTAL_DURATION * pxPerSecond;
    area.style.width = totalWidth + 'px';

    const showWaiting = document.getElementById('showWaiting').checked;

    // Create agent rows
    const agentRows = {{}};
    AGENTS.forEach((agent, idx) => {{
        const row = document.createElement('div');
        row.className = 'agent-row';
        row.style.width = totalWidth + 'px';
        area.appendChild(row);
        agentRows[agent] = row;
    }});

    // Place task blocks
    TASKS.forEach(t => {{
        const row = agentRows[t.agent];
        if (!row) return;

        const active = activeCases.has(t.caseId);

        // Waiting time block (hatched)
        if (showWaiting && t.waiting > 1 && active) {{
            const wb = document.createElement('div');
            wb.className = 'waiting-block';
            wb.style.left = ((t.start - t.waiting) * pxPerSecond) + 'px';
            wb.style.width = (t.waiting * pxPerSecond) + 'px';
            row.appendChild(wb);
        }}

        // Task block
        const block = document.createElement('div');
        block.className = 'task-block' + (t.isCollab ? ' collab' : '') + (!active ? ' dimmed' : '');
        block.style.left = (t.start * pxPerSecond) + 'px';
        block.style.width = Math.max(2, (t.end - t.start) * pxPerSecond) + 'px';
        block.style.background = caseColor(t.caseId);
        block.style.zIndex = active ? 10 : 1;

        // Label if wide enough
        const blockWidth = (t.end - t.start) * pxPerSecond;
        if (blockWidth > 30) {{
            block.textContent = t.label;
        }}

        // Tooltip
        block.addEventListener('mouseenter', e => showTooltip(e, t));
        block.addEventListener('mousemove', e => moveTooltip(e));
        block.addEventListener('mouseleave', hideTooltip);

        row.appendChild(block);
    }});
}}

// Tooltip
const tooltipEl = document.getElementById('tooltip');

function showTooltip(e, t) {{
    const dur = ((t.end - t.start) / 60).toFixed(1);
    const wait = (t.waiting / 60).toFixed(1);
    tooltipEl.innerHTML = `
        <div class="label">${{t.label}}: ${{t.taskName}}</div>
        <div class="detail">
            Agent: ${{t.agent}}<br>
            ${{t.isCollab ? 'Collab with: ' + t.allAgents + '<br>' : ''}}
            Duration: ${{dur}} min<br>
            Waiting: ${{wait}} min<br>
            Case: ${{t.caseId}} &middot; Task ID: ${{t.taskId}}
        </div>
    `;
    tooltipEl.classList.add('visible');
    moveTooltip(e);
}}

function moveTooltip(e) {{
    let x = e.clientX + 12;
    let y = e.clientY + 12;
    const rect = tooltipEl.getBoundingClientRect();
    if (x + 320 > window.innerWidth) x = e.clientX - 320;
    if (y + 200 > window.innerHeight) y = e.clientY - 200;
    tooltipEl.style.left = x + 'px';
    tooltipEl.style.top = y + 'px';
}}

function hideTooltip() {{
    tooltipEl.classList.remove('visible');
}}

// Zoom
const zoomSlider = document.getElementById('zoomSlider');
const zoomLabel = document.getElementById('zoomLabel');

zoomSlider.addEventListener('input', () => {{
    pxPerSecond = parseFloat(zoomSlider.value);
    zoomLabel.textContent = pxPerSecond.toFixed(1) + ' px/s';
    renderTimeAxis();
    renderTasks();
}});

function resetZoom() {{
    pxPerSecond = 3;
    zoomSlider.value = 3;
    zoomLabel.textContent = '3.0 px/s';
    renderTimeAxis();
    renderTasks();
}}

function fitAll() {{
    const wrapper = document.getElementById('wrapper');
    const available = wrapper.clientWidth - 160;
    pxPerSecond = Math.max(0.5, available / TOTAL_DURATION);
    zoomSlider.value = Math.min(20, pxPerSecond);
    zoomLabel.textContent = pxPerSecond.toFixed(1) + ' px/s';
    renderTimeAxis();
    renderTasks();
}}

document.getElementById('showWaiting').addEventListener('change', renderTasks);

// Initial render
renderTimeAxis();
renderTasks();
</script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description='Generate interactive agent task timeline')
    parser.add_argument('log_csv', type=str, help='Path to log CSV file')
    parser.add_argument('--max-cases', type=int, default=None, help='Max number of cases to show')
    parser.add_argument('--output', type=str, default=None, help='Output HTML file path')
    parser.add_argument('--title', type=str, default=None, help='Timeline title')
    args = parser.parse_args()

    csv_path = Path(args.log_csv)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return

    print(f"Loading: {csv_path}")
    tasks, agents, case_ids = load_log(csv_path, max_cases=args.max_cases)
    print(f"  {len(tasks)} tasks, {len(agents)} agents, {len(case_ids)} cases")

    title = args.title or f"Timeline: {csv_path.stem}"
    html = generate_html(tasks, agents, case_ids, title=title)

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('analysis') / f"timeline_{csv_path.stem}.html"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Timeline saved: {output_path}")
    print(f"Open in browser: file://{output_path.resolve()}")


if __name__ == '__main__':
    main()
