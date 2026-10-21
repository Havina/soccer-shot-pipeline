/* Soccer ML frontend — vanilla JS, talks only to the FastAPI backend in app/main.py. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const ACTIVE_STATES = new Set(["QUEUED", "RUNNING", "CONFIRMING"]);
  const POLL_MS = 3000;

  let selectedFile = null;
  let currentJobId = null;
  let pollTimer = null;
  let pickerOpenFor = null;      // job id whose picker is open via "Change shooter"
  let previewFrame = null;       // frame currently shown in the preview image
  const CANDIDATE_COLORS = ["#50dc64", "#3caaff", "#ff5a5a", "#ffc800", "#e65aff", "#78ffff"];

  // ------------------------------------------------------------------ helpers
  async function api(path, options = {}) {
    const res = await fetch(path, options);
    let body = null;
    try { body = await res.json(); } catch (_) { /* no body */ }
    if (!res.ok) {
      const detail = body && body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : res.statusText;
      throw new Error(detail);
    }
    return body;
  }

  const fmt = {
    m: (v) => (v == null ? "—" : `${v.toFixed(2)} m`),
    deg: (v) => (v == null ? "—" : `${v.toFixed(1)}°`),
    pct: (v) => (v == null ? "—" : `${v.toFixed(1)}%`),
    int: (v) => (v == null ? "—" : String(v)),
    str: (v) => (v == null ? "—" : String(v)),
    when: (iso) => { try { return new Date(iso).toLocaleString(); } catch (_) { return iso; } },
  };

  function pillClass(state) {
    if (ACTIVE_STATES.has(state)) return "running";
    if (state === "AWAITING_SHOOTER_CONFIRMATION") return "waiting";
    if (state === "COMPLETE") return "complete";
    if (state === "FAILED") return "failed";
    return "";
  }

  function stateLabel(state) {
    return {
      QUEUED: "Queued",
      RUNNING: "Analyzing…",
      AWAITING_SHOOTER_CONFIRMATION: "Needs your confirmation",
      CONFIRMING: "Replaying with confirmed shooter…",
      COMPLETE: "Complete",
      FAILED: "Failed",
    }[state] || state;
  }

  // ------------------------------------------------------------------ health
  async function refreshHealth() {
    const el = $("health"), txt = $("health-text");
    try {
      const h = await api("/health");
      el.className = "health " + (h.status === "READY" ? "health-ok" : h.status === "READY_UNPROBED" ? "health-warn" : "health-bad");
      txt.textContent = h.status === "READY" ? `Ready · ${h.runtime.gpu_name || "local GPU"}`
        : h.status === "READY_UNPROBED" ? "Ready (runtime unprobed)"
        : h.status === "RUNTIME_NOT_READY" ? `Runtime not ready: ${h.runtime.failure_reason || ""}`
        : `Integrity failure: ${h.integrity.mismatches.join("; ")}`;
      $("analyze-btn").disabled = !(h.can_analyze && selectedFile);
      el.title = `certification: ${h.certification_kind}\nfingerprint: ${h.integrity.certified_runtime_fingerprint_sha256 || "?"}`;
    } catch (err) {
      el.className = "health health-bad";
      txt.textContent = "Backend unreachable";
      $("analyze-btn").disabled = true;
    }
  }

  // ------------------------------------------------------------------ upload
  function setFile(file) {
    selectedFile = file || null;
    $("file-name").textContent = file ? `${file.name} · ${(file.size / 1048576).toFixed(1)} MB` : "No file selected";
    $("upload-error").classList.add("hidden");
    refreshHealth();
  }

  function uploadWithProgress(file) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const form = new FormData();
      form.append("file", file, file.name);
      const prog = $("upload-progress"), bar = prog.querySelector(".bar"), label = prog.querySelector(".label");
      prog.classList.remove("hidden");
      xhr.upload.onprogress = (e) => {
        if (!e.lengthComputable) return;
        const p = Math.round((e.loaded / e.total) * 100);
        bar.style.width = p + "%";
        label.textContent = p < 100 ? `Uploading ${p}%` : "Starting analysis…";
      };
      xhr.onload = () => {
        prog.classList.add("hidden"); bar.style.width = "0";
        let body = null;
        try { body = JSON.parse(xhr.responseText); } catch (_) { /* ignore */ }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new Error(body && body.detail ? body.detail : `Upload failed (${xhr.status})`));
      };
      xhr.onerror = () => { prog.classList.add("hidden"); reject(new Error("Network error during upload")); };
      xhr.open("POST", "/analyze");
      xhr.send(form);
    });
  }

  async function startAnalysis() {
    if (!selectedFile) return;
    const btn = $("analyze-btn");
    btn.disabled = true;
    $("upload-error").classList.add("hidden");
    try {
      const job = await uploadWithProgress(selectedFile);
      showJob(job);
      startPolling(job.job_id);
      refreshJobList();
    } catch (err) {
      $("upload-error").textContent = err.message;
      $("upload-error").classList.remove("hidden");
    } finally {
      btn.disabled = !selectedFile;
    }
  }

  // ------------------------------------------------------------------ job rendering
  function showJob(job) {
    currentJobId = job.job_id;
    $("job-card").classList.remove("hidden");
    $("job-id").textContent = job.job_id;
    $("job-file").textContent = job.original_filename;
    const pill = $("job-state");
    pill.textContent = stateLabel(job.state);
    pill.className = "pill " + pillClass(job.state);

    const summary = job.confirmed || job.automatic;
    const stages = summary ? summary.completed_stage_count : (job.state === "RUNNING" ? 0 : 0);
    const pct = Math.round((stages / 18) * 100);
    $("stage-bar").style.width = pct + "%";
    $("stage-text").textContent = summary
      ? `${stages} / 18 stages${summary.blocked_stage ? ` · waiting at ${summary.blocked_stage}` : ""}`
      : (ACTIVE_STATES.has(job.state) ? "Running the 18‑stage pipeline…" : "0 / 18 stages");

    renderPreprocess(job);
    const errEl = $("job-error");
    if (job.error) { errEl.textContent = job.error; errEl.classList.remove("hidden"); } else { errEl.classList.add("hidden"); }

    renderConfirmation(job);
    renderContextForm(job);
    renderResults(job);
    highlightJobInList(job.job_id);
  }

  function renderPreprocess(job) {
    const box = $("preprocess-box");
    const p = job.preprocess;
    if (!p) { box.classList.add("hidden"); box.innerHTML = ""; return; }
    const bits = [];
    if (p.width && p.height) bits.push(`${p.width}×${p.height}`);
    if (p.duration_s) bits.push(`${p.duration_s.toFixed(1)}s`);
    if (p.fps) bits.push(`${Math.round(p.fps)} fps`);
    if (p.trimmed) bits.push("auto-trimmed");
    if ((p.cut_frames || []).length) bits.push(`${p.cut_frames.length} camera cut${p.cut_frames.length === 1 ? "" : "s"}`);
    const overlays = (p.overlays || []).map((o) => `<span class="overlay">${o}</span>`).join("");
    const warns = (p.warnings || []).map((w) => `<li>${w}</li>`).join("");
    const geo = p.geometry_fallback && p.geometry_fallback.used
      ? `<p class="hint">Pitch pose taken from nearby frame ${p.geometry_fallback.fallback_frame_idx} (offset ${p.geometry_fallback.offset}). Not Cell‑40 certified.</p>`
      : "";
    const team = p.team_assignment_v3 && (p.team_assignment_v3.ties_broken || p.team_assignment_v3.short_tracks_ignored)
      ? `<p class="hint">Team overlay broke ${p.team_assignment_v3.ties_broken || 0} colour tie(s) and ignored ${p.team_assignment_v3.short_tracks_ignored || 0} short track(s).</p>`
      : "";
    const goal = p.goal_end && p.goal_end.flipped
      ? `<p class="hint">${p.goal_end.reason || "Flipped to the near pitch end. Not Cell‑40 certified."}</p>`
      : "";
    box.innerHTML = `<div>${bits.map((b) => `<span class="k">${b}</span>`).join(" · ")}</div>` +
      (overlays ? `<div style="margin-top:.35rem">${overlays}</div>` : "") +
      geo + team + goal +
      (warns ? `<ul>${warns}</ul>` : "");
    box.classList.remove("hidden");
  }

  // ------------------------------------------------------------------ shooter picker
  function teamLabel(job, side) {
    const ctx = job.match_context || {};
    const name = side === "ATTACKING" ? ctx.attacking_team : side === "DEFENDING" ? ctx.defending_team : null;
    const color = side === "ATTACKING" ? ctx.attacking_color : side === "DEFENDING" ? ctx.defending_color : null;
    if (!name && !color) return side || "—";
    return `${name || side}${color ? ` (${color})` : ""}`;
  }

  function renderConfirmation(job) {
    const box = $("confirm-box");
    const needsPick = job.state === "AWAITING_SHOOTER_CONFIRMATION";
    const changing = pickerOpenFor === job.job_id && job.can_change_shooter && !ACTIVE_STATES.has(job.state);
    if (!(needsPick || changing) || !job.automatic) { box.classList.add("hidden"); return; }
    box.classList.remove("hidden");
    $("confirm-title").textContent = changing ? "Change the shooter" : "Who took the shot?";
    $("confirm-text").textContent = changing
      ? `Currently using Track ${job.confirmed && job.confirmed.shooter.resolved_track_id != null ? job.confirmed.shooter.resolved_track_id : "—"}. Pick a different player to replay the analysis with that shooter.`
      : "The automatic ranking could not separate these players, so the pipeline stopped and is asking you. Look at the frame and pick the player who actually struck the ball.";
    $("confirm-close").classList.toggle("hidden", !changing);

    // preview frame
    const img = $("preview-img"), missing = $("preview-missing"), nav = $("preview-nav");
    if (job.links.candidates_preview) {
      if (previewFrame == null || img.dataset.jobId !== job.job_id) previewFrame = job.shot_frame_idx;
      img.dataset.jobId = job.job_id;
      setPreviewFrame(job, previewFrame);
      img.classList.remove("hidden"); missing.classList.add("hidden"); nav.classList.remove("hidden");
    } else {
      img.classList.add("hidden"); missing.classList.remove("hidden"); nav.classList.add("hidden");
    }

    // candidate buttons
    const wrap = $("candidates");
    wrap.innerHTML = "";
    const current = job.confirmed ? job.confirmed.shooter.resolved_track_id : null;
    job.automatic.shooter.candidates.forEach((c, i) => {
      const b = document.createElement("button");
      b.className = "candidate";
      b.style.borderColor = CANDIDATE_COLORS[i % CANDIDATE_COLORS.length];
      const isCurrent = current != null && c.track_id === current;
      b.disabled = isCurrent;
      b.innerHTML =
        (c.crop_url ? `<img src="${c.crop_url}" alt="Track ${c.track_id}" loading="lazy" />` : `<span class="swatch" style="background:${CANDIDATE_COLORS[i % CANDIDATE_COLORS.length]}"></span>`) +
        `<span><span class="track">Track ${c.track_id}</span>` +
        `<span class="score">auto score ${c.auto_score == null ? "—" : c.auto_score.toFixed(3)}` +
        `${c.contact_frame_idx != null ? ` · contact ~frame ${c.contact_frame_idx}` : ""}</span>` +
        `<span class="verb">${isCurrent ? "current shooter" : "This player took the shot"}</span></span>`;
      b.onclick = () => confirmShooter(job.job_id, c.track_id, wrap);
      wrap.appendChild(b);
    });
  }

  function setPreviewFrame(job, frame) {
    previewFrame = frame;
    const img = $("preview-img");
    const base = job.links.candidates_preview;
    img.src = frame == null ? base : `${base}?frame=${Math.max(0, frame)}`;
    $("preview-frame").textContent = frame == null ? "shot frame" : `frame ${frame}${frame === job.shot_frame_idx ? " (estimated shot moment)" : ""}`;
  }

  async function confirmShooter(jobId, trackId, wrap) {
    for (const b of wrap.querySelectorAll("button")) b.disabled = true;
    try {
      const job = await api(`/jobs/${jobId}/confirm`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ shooter_track_id: trackId }),
      });
      pickerOpenFor = null;
      showJob(job);
      startPolling(jobId);
    } catch (err) {
      $("job-error").textContent = err.message;
      $("job-error").classList.remove("hidden");
      for (const b of wrap.querySelectorAll("button")) b.disabled = false;
    }
  }

  // ------------------------------------------------------------------ team labels (display only)
  function renderContextForm(job) {
    const form = $("context-form");
    const ctx = job.match_context || {};
    for (const el of form.elements) if (el.name) el.value = ctx[el.name] || "";
    form.dataset.jobId = job.job_id;
  }

  async function saveContext(e) {
    e.preventDefault();
    const form = $("context-form"), jobId = form.dataset.jobId;
    if (!jobId) return;
    const body = {};
    for (const el of form.elements) if (el.name) body[el.name] = el.value.trim() || null;
    try {
      const job = await api(`/jobs/${jobId}/context`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      $("context-saved").classList.remove("hidden");
      setTimeout(() => $("context-saved").classList.add("hidden"), 1500);
      showJob(job);
    } catch (err) {
      $("job-error").textContent = err.message; $("job-error").classList.remove("hidden");
    }
  }

  function metric(k, v) { return `<div class="metric"><span class="k">${k}</span><span class="v">${v}</span></div>`; }

  function renderResults(job) {
    const card = $("results-card");
    const summary = job.confirmed || (job.automatic && job.automatic.semantic_state === "COMPLETE" ? job.automatic : null);
    if (!summary) { card.classList.add("hidden"); $("geometry-warning").classList.add("hidden"); return; }
    card.classList.remove("hidden");
    const m = summary.metrics;
    $("xg-value").textContent = fmt.pct(m.xg_percent);
    const otherSide = m.shooter_relative_team === "ATTACKING" ? "DEFENDING" : m.shooter_relative_team === "DEFENDING" ? "ATTACKING" : null;
    const goalNote = m.target_goal === "LEFT" ? "pitch x=0" : m.target_goal === "RIGHT" ? "pitch x=105" : "";
    $("metrics").innerHTML = [
      metric("Shooter", m.shooter_track_id == null ? "—" : `Track ${m.shooter_track_id}${summary.shooter.manual_confirmation_used ? " (confirmed by you)" : " (automatic)"}`),
      metric("Shooter's team", teamLabel(job, m.shooter_relative_team)),
      metric("Defending team", otherSide ? teamLabel(job, otherSide) : "—"),
      metric("Target goal", m.target_goal == null ? "—" : `${m.target_goal}${goalNote ? ` (${goalNote})` : ""}`),
      metric("Distance to goal", fmt.m(m.distance_to_goal_m)),
      metric("Near pitch end", m.distance_to_left_goal_m == null || m.distance_to_right_goal_m == null ? "—" :
        (m.distance_to_left_goal_m <= m.distance_to_right_goal_m
          ? `LEFT ${fmt.m(m.distance_to_left_goal_m)}`
          : `RIGHT ${fmt.m(m.distance_to_right_goal_m)}`)),
      metric("Far pitch end", m.distance_to_left_goal_m == null || m.distance_to_right_goal_m == null ? "—" :
        (m.distance_to_left_goal_m > m.distance_to_right_goal_m
          ? `LEFT ${fmt.m(m.distance_to_left_goal_m)}`
          : `RIGHT ${fmt.m(m.distance_to_right_goal_m)}`)),
      metric("Shot angle", fmt.deg(m.shot_angle_deg)),
      metric("Nearest defender", fmt.m(m.nearest_defender_m)),
      metric("Defenders within 5 m", fmt.int(m.defenders_within_5m)),
      metric("xG probability", m.xg_probability == null ? "—" : m.xg_probability.toFixed(4)),
    ].join("");
    const warn = $("geometry-warning");
    if (m.geometry_warnings && m.geometry_warnings.length) {
      warn.textContent = m.geometry_warnings.join(" ");
      warn.classList.remove("hidden");
    } else { warn.classList.add("hidden"); warn.textContent = ""; }

    const video = $("annotated-video"), missing = $("video-missing"), dl = $("download-link"), js = $("json-link");
    if (job.links.annotated_video) {
      const src = job.links.annotated_video;
      if (video.getAttribute("data-src") !== src) { video.src = src; video.setAttribute("data-src", src); }
      video.classList.remove("hidden"); missing.classList.add("hidden");
      dl.href = src; dl.classList.remove("hidden");
    } else {
      video.classList.add("hidden"); missing.classList.remove("hidden"); dl.classList.add("hidden");
    }
    js.href = job.links.result_json; js.classList.remove("hidden");

    const change = $("change-shooter");
    change.classList.toggle("hidden", !(job.can_change_shooter && !ACTIVE_STATES.has(job.state)));
    change.onclick = () => { pickerOpenFor = job.job_id; previewFrame = null; showJob(job); $("confirm-box").scrollIntoView({ behavior: "smooth", block: "start" }); };

    const hist = $("confirmation-history");
    if (job.confirmations && job.confirmations.length) {
      hist.textContent = "Confirmations: " + job.confirmations.map((c) => `Track ${c.shooter_track_id} → ${c.outcome || "pending"} (${fmt.when(c.at)})`).join(" · ");
      hist.classList.remove("hidden");
    } else { hist.classList.add("hidden"); }

    const g = summary.geometry;
    $("geometry").innerHTML = [
      ["Selected frame", fmt.int(g.selected_frame_idx)],
      ["Calibration variant", `${fmt.str(g.selected_variant)} · ${g.validation_status}`],
      ["Keypoints / lines", `${fmt.int(g.keypoint_count)} / ${fmt.int(g.line_count)}`],
      ["Reference‑free", g.reference_free ? "yes" : "no"],
      ["Runtime profile", fmt.str(g.runtime_profile)],
      ["Runtime fingerprint", fmt.str(g.runtime_fingerprint_sha256)],
      ["Public API calls", fmt.int(job.public_api_calls)],
      ["Video SHA‑256", fmt.str(job.video_sha256)],
    ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  }

  // ------------------------------------------------------------------ polling
  function startPolling(jobId) {
    stopPolling();
    const tick = async () => {
      try {
        const job = await api(`/jobs/${jobId}`);
        if (currentJobId === jobId) showJob(job);
        if (ACTIVE_STATES.has(job.state)) pollTimer = setTimeout(tick, POLL_MS);
        else refreshJobList();
      } catch (err) {
        pollTimer = setTimeout(tick, POLL_MS * 2);
      }
    };
    pollTimer = setTimeout(tick, 1000);
  }
  function stopPolling() { if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; } }

  // ------------------------------------------------------------------ job list
  async function refreshJobList() {
    const list = $("job-list");
    try {
      const jobs = await api("/jobs");
      list.innerHTML = "";
      if (!jobs.length) { list.innerHTML = '<li class="hint">No jobs yet.</li>'; return; }
      for (const j of jobs) {
        const li = document.createElement("li");
        const b = document.createElement("button");
        b.className = "job-item" + (j.job_id === currentJobId ? " active" : "");
        b.dataset.jobId = j.job_id;
        b.innerHTML = `<span class="name">${j.original_filename}<span class="when">${fmt.when(j.created_at)}</span></span>` +
          `<span class="pill ${pillClass(j.state)}">${stateLabel(j.state)}</span>`;
        b.onclick = async () => { const job = await api(`/jobs/${j.job_id}`); showJob(job); if (ACTIVE_STATES.has(job.state)) startPolling(job.job_id); else stopPolling(); };
        li.appendChild(b);
        list.appendChild(li);
      }
    } catch (_) { /* leave list as-is */ }
  }
  async function clearJobs() {
    let jobs;
    try { jobs = await api("/jobs"); } catch (_) { return; }
    const finished = jobs.filter((j) => !ACTIVE_STATES.has(j.state));
    const running = jobs.length - finished.length;
    if (!finished.length) { alert(running ? "All jobs are still running; nothing to clear yet." : "Match sheet is already empty."); return; }
    const msg = `Delete ${finished.length} finished job${finished.length === 1 ? "" : "s"} and their videos/results?` +
      (running ? `\n(${running} running job${running === 1 ? "" : "s"} will be kept.)` : "");
    if (!confirm(msg)) return;
    const btn = $("clear-jobs");
    btn.disabled = true;
    try {
      for (const j of finished) {
        try { await api(`/jobs/${j.job_id}`, { method: "DELETE" }); } catch (_) { /* skip and continue */ }
        if (j.job_id === currentJobId) {
          stopPolling(); currentJobId = null; pickerOpenFor = null;
          $("job-card").classList.add("hidden"); $("results-card").classList.add("hidden");
        }
      }
    } finally {
      btn.disabled = false;
      refreshJobList();
    }
  }
  function highlightJobInList(jobId) {
    for (const b of document.querySelectorAll(".job-item")) b.classList.toggle("active", b.dataset.jobId === jobId);
  }

  // ------------------------------------------------------------------ wiring
  const dz = $("dropzone"), input = $("file-input");
  $("browse-btn").onclick = () => input.click();
  dz.onclick = (e) => { if (e.target === dz || e.target.id === "drop-text") input.click(); };
  input.onchange = () => setFile(input.files[0]);
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) setFile(f); });
  $("analyze-btn").onclick = startAnalysis;
  $("refresh-jobs").onclick = refreshJobList;
  $("clear-jobs").onclick = clearJobs;
  $("context-form").onsubmit = saveContext;
  $("confirm-close").onclick = async () => { pickerOpenFor = null; if (currentJobId) showJob(await api(`/jobs/${currentJobId}`)); };
  for (const b of document.querySelectorAll("#preview-nav button")) {
    b.onclick = async () => {
      if (!currentJobId) return;
      const job = await api(`/jobs/${currentJobId}`);
      const base = previewFrame == null ? (job.shot_frame_idx || 0) : previewFrame;
      setPreviewFrame(job, Math.max(0, base + parseInt(b.dataset.step, 10)));
    };
  }

  refreshHealth();
  refreshJobList();
  setInterval(refreshHealth, 15000);
})();
