# Nginx config flatten — migration guide

**Affects:** `analyzer/services/project_provisioning_service.py`
**Version landed:** dev container rebuild on **2026-04-23** (not yet in a tagged release).
**Prod status:** **NOT YET DEPLOYED.** Prod container still runs the pre-v3.10 image and will hit the legacy-path fallback (see §6).

---

## 1. Why this change exists

The host server enforces a "ONE `docker-compose.yml` + ONE `nginx.conf`" rule. Previously, paperless-ai-analyzer wrote per-project include files to `/etc/nginx/projects-locations.d/*.conf` which nginx consumed via `include projects-locations.d/*.conf;` inside the main 443 server block. That was technically "one logical config" but it violated the literal single-file rule.

The host nginx.conf has been flattened. The projects-locations.d directory is emptied and the include directive is removed. A pair of sentinel markers have been added inside the 443 server block instead:

```nginx
# ═══════════════════════════════════════════════════════════════════
# AUTO-MANAGED by paperless-ai-analyzer — do not hand-edit inside this
# block. paperless-ai-analyzer reads between the sentinel markers and
# rewrites them when projects are created/deleted, then reloads nginx.
# Manual edits inside this block WILL be overwritten.
# ═══════════════════════════════════════════════════════════════════
# <paperless-projects-begin>
# <paperless-project slug="pw-flow-04222114">
        location /paperless-pw-flow-04222114/ { ... }
        location = /paperless-pw-flow-04222114 { ... }
# </paperless-project>
# <paperless-project slug="pw-reg-04222114">
        ...
# </paperless-project>
# <paperless-projects-end>
```

Each project's block is wrapped in its own nested markers (`# <paperless-project slug="SLUG">` / `# </paperless-project>`) so individual projects can be inserted, replaced, or removed by string-substituting between their own markers.

## 2. Code changes

Two new helpers in `analyzer/services/project_provisioning_service.py`:

- **`_update_nginx_project_block(slug, block_content) -> bool`** — opens `/app/nginx.conf` with `fcntl.LOCK_EX`, locates the outer sentinels, and inserts or replaces this project's sub-block. Returns `True` on success, `False` if the sentinels aren't present (legacy deployments).
- **`_remove_nginx_project_block(slug) -> bool`** — same locking pattern; removes the sub-block if present.

The provisioning path (`_provision_project_paperless`) is modified as follows:

```python
# New (v3.10+):
if not _update_nginx_project_block(slug, nginx_conf):
    # Sentinels missing (not-yet-flattened environment). Fall back to legacy dir write.
    nginx_conf_path = f'/app/nginx-projects-locations.d/paperless-{slug}.conf'
    with open(nginx_conf_path, 'w') as f:
        f.write(nginx_conf)
```

**Forward AND backward compatible.** Same code works on flattened and unflattened hosts; sentinel detection decides the path.

## 3. docker-compose.yml mount changes

Each paperless-ai-analyzer variant needs **two** mounts, not one, during rollout:

```yaml
volumes:
- ./config/nginx/nginx.conf:/app/nginx.conf:rw                    # new — primary path
- ./config/nginx/projects-locations.d:/app/nginx-projects-locations.d:rw   # legacy fallback
```

Once all environments have flattened and no container is running pre-v3.10 code anywhere, the legacy dir mount can be removed and the dir deleted from the host.

**Currently updated:** `paperless-ai-analyzer-dev` only.
**Needs update on next deploy:** `paperless-ai-analyzer` (prod), `paperless-ai-analyzer-jacob`.

## 4. Project deletion (existing gap)

`project_manager.delete_project` never cleaned up nginx config — even in the pre-flatten code. A new project's location block lived forever after deletion. The helper `_remove_nginx_project_block` is added for the day `delete_project` is wired to call it. **Not part of this change** — left as a follow-up so reviewers can see the flatten diff cleanly.

## 5. Prod deploy checklist (when ready)

1. Pull updated image (contains `_update_nginx_project_block` + `_remove_nginx_project_block`).
2. Update `docker-compose.yml` for `paperless-ai-analyzer` service:
   - Add the `./config/nginx/nginx.conf:/app/nginx.conf:rw` mount.
   - Leave the legacy dir mount for one more release cycle (can remove on the release after).
3. `docker compose up -d --force-recreate --no-deps paperless-ai-analyzer`
4. Verify an existing project still loads: `curl -skI https://www.voipguru.org/paperless-pw-reg-04222114/ | head -3` → 301 or 302 expected.
5. Create a new test project from the UI. Confirm:
   - A `# <paperless-project slug="...">` sub-block appears between the sentinels in `/etc/nginx/nginx.conf` on the host.
   - `sudo docker exec nginx nginx -s reload` succeeded in the container logs.
   - The new project URL returns 200/302.
6. After first successful prod project creation via new path, the rollout is complete for that container.

Same procedure for `paperless-ai-analyzer-jacob`.

## 6. Rollback

Two paths, depending on how far rollout got:

**Soft rollback (code only):**
1. Pin the prod image tag to the pre-v3.10 version.
2. `docker compose up -d --force-recreate --no-deps paperless-ai-analyzer`
3. That container reverts to the legacy dir-write path. But note: `/etc/nginx/projects-locations.d/*.conf` include is **no longer in nginx.conf**. Fresh projects written by the legacy code will go unread by nginx until…

**Full rollback (code + nginx.conf):**
1. Pin image to pre-v3.10.
2. Restore the old nginx.conf: `cp /home/dblagbro/backups/consolidation-20260423-194041/nginx.conf.pre-flatten.bak /home/dblagbro/docker/config/nginx/nginx.conf`
3. `docker exec nginx nginx -s reload`
4. Projects-locations.d directory is still populated from the flatten migration (we kept it). System is back to the pre-flatten topology.

## 7. Testing

Minimum tests before shipping to prod:

- [ ] **Idempotent re-provision**: call `_update_nginx_project_block('test-abc', block)` twice with different `block` content. Confirm only one `# <paperless-project slug="test-abc">` section ends up in nginx.conf, with the latest body.
- [ ] **Concurrent provisioning**: spawn two threads each calling `_update_nginx_project_block` with different slugs. Both must succeed; final nginx.conf must contain both blocks. (Exercised by `fcntl.flock` in `_update_nginx_project_block`.)
- [ ] **Sentinel-missing fallback**: temporarily remove the markers from nginx.conf, confirm `_update_nginx_project_block` returns `False` and the caller falls back to the dir-write path.
- [ ] **Remove path**: call `_remove_nginx_project_block('test-abc')`. Confirm the block is gone from nginx.conf.
- [ ] **Nginx reload**: after every update, `docker exec nginx nginx -s reload` must succeed. Any syntax error in our generated block would break ALL of nginx, not just paperless. The helper deliberately writes raw text without quoting — we trust the template string. If you ever change the template, run `nginx -t` manually first.

## 8. Security / correctness notes

- `/app/nginx.conf` is mounted **rw** into every analyzer container. That means ANY remote-code-execution bug in analyzer → attacker can rewrite nginx.conf → nginx reload → full server takeover. This risk is unchanged from the previous per-file setup (the directory was also rw-mounted), but flattening makes the blast radius visible: a single file edit vs. a per-project file.
- `fcntl.flock` serializes concurrent writes from the SAME container. Cross-container serialization (dev vs. prod vs. jacob all writing at once) is NOT protected — different processes, different file descriptors, different locks. In practice this is fine because provisioning is per-instance (each container has its own project set), but worth noting.
- The helper does not validate the `block_content`. An attacker who controls slug→block would have free nginx-config write. Mitigated by paperless-ai-analyzer's existing slug sanitization — no changes here.
