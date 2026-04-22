// ── User Management (admin) ─────────────────────────────────────────────

        // ── User management (admin) ────────────────────────────────────
        async function loadUsers() {
            try {
                const res = await apiFetch(apiUrl('/api/users'));
                if (!res.ok) return;
                const data = await res.json();
                renderUsersTable(data.users);
            } catch (e) { console.error(e); }
        }

        function renderUsersTable(users) {
            const tbody = document.getElementById('users-table-body');
            if (!tbody) return;
            if (!users || !users.length) {
                tbody.innerHTML = '<tr><td colspan="7" style="color:#999;">No users found</td></tr>';
                return;
            }
            tbody.innerHTML = users.map(u => `<tr>
                <td><strong>${escapeHtml(u.username)}</strong></td>
                <td>${escapeHtml(u.display_name || '')}</td>
                <td style="font-size:12px; color:#555;">${escapeHtml(u.email || '—')}</td>
                <td><span class="role-badge ${u.role}">${u.role}</span></td>
                <td style="font-size:12px; color:#666;">${u.last_login ? u.last_login.split('T')[0] : 'Never'}</td>
                <td>${u.is_active ? '✅' : '❌'}</td>
                <td style="display:flex; gap:6px; flex-wrap:wrap;">
                    <button class="btn" style="font-size:11px; padding:3px 8px;" onclick="openEditUserModal(${JSON.stringify(u).replace(/"/g,'&quot;')})">✏️ Edit</button>
                    ${u.is_active
                        ? `<button class="btn btn-danger" style="font-size:11px; padding:3px 8px;" onclick="deactivateUser(${u.id})">Deactivate</button>`
                        : `<button class="btn btn-success" style="font-size:11px; padding:3px 8px;" onclick="activateUser(${u.id})">Activate</button>`
                    }
                </td>
            </tr>`).join('');
        }

        function showAddUserForm() {
            document.getElementById('add-user-form').style.display = 'block';
        }
        function hideAddUserForm() {
            document.getElementById('add-user-form').style.display = 'none';
        }

        async function createUser() {
            const username = document.getElementById('new-user-username').value.trim();
            const password = document.getElementById('new-user-password').value.trim();
            const display_name = document.getElementById('new-user-display').value.trim();
            const email = document.getElementById('new-user-email').value.trim();
            const role = document.getElementById('new-user-role').value;
            const errEl = document.getElementById('add-user-error');
            if (!username || !password) { errEl.textContent = 'Username and password required'; errEl.style.display = 'block'; return; }
            errEl.style.display = 'none';
            try {
                const res = await apiFetch(apiUrl('/api/users'), {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({username, password, role, display_name, email}),
                });
                const data = await res.json();
                if (!res.ok) { errEl.textContent = data.error || 'Failed to create user'; errEl.style.display = 'block'; return; }
                hideAddUserForm();
                ['new-user-username','new-user-password','new-user-display','new-user-email'].forEach(id => {
                    const el = document.getElementById(id); if (el) el.value = '';
                });
                if (data.email_sent) alert(`✅ User created and welcome email sent to ${email}`);
                loadUsers();
            } catch (e) { errEl.textContent = e.message; errEl.style.display = 'block'; }
        }

        function openEditUserModal(u) {
            document.getElementById('edit-user-id').value = u.id;
            document.getElementById('edit-user-name-label').textContent = u.username;
            document.getElementById('edit-display-name').value = u.display_name || '';
            document.getElementById('edit-job-title').value = u.job_title || '';
            document.getElementById('edit-email').value = u.email || '';
            document.getElementById('edit-phone').value = u.phone || '';
            document.getElementById('edit-address').value = u.address || '';
            document.getElementById('edit-github').value = u.github || '';
            document.getElementById('edit-linkedin').value = u.linkedin || '';
            document.getElementById('edit-facebook').value = u.facebook || '';
            document.getElementById('edit-instagram').value = u.instagram || '';
            document.getElementById('edit-other-handles').value = u.other_handles || '';
            document.getElementById('edit-role').value = u.role || 'basic';
            document.getElementById('edit-timezone').value = u.timezone || '';
            document.getElementById('edit-password').value = '';
            document.getElementById('edit-user-error').style.display = 'none';
            const modal = document.getElementById('edit-user-modal');
            modal.style.display = 'flex';
        }

        function closeEditUserModal() {
            document.getElementById('edit-user-modal').style.display = 'none';
        }

        async function emailUserManual() {
            const uid = document.getElementById('edit-user-id').value;
            const errEl = document.getElementById('edit-user-error');
            const okEl  = document.getElementById('edit-user-success');
            errEl.style.display = 'none';
            okEl.style.display  = 'none';
            try {
                const r = await fetch(`${window.APP_CONFIG.basePath}/api/users/${uid}/send-manual`, {method:'POST'});
                const d = await r.json();
                if (!r.ok) throw new Error(d.error || 'Failed');
                okEl.textContent = `✓ ${d.message}`;
                okEl.style.display = 'block';
                setTimeout(() => { okEl.style.display = 'none'; }, 4000);
            } catch(e) {
                errEl.textContent = `Failed to send: ${e.message}`;
                errEl.style.display = 'block';
            }
        }

        async function saveEditUser() {
            const uid = document.getElementById('edit-user-id').value;
            const display_name = document.getElementById('edit-display-name').value.trim();
            const job_title = document.getElementById('edit-job-title').value.trim();
            const email = document.getElementById('edit-email').value.trim();
            const phone = document.getElementById('edit-phone').value.trim();
            const address = document.getElementById('edit-address').value.trim();
            const github = document.getElementById('edit-github').value.trim();
            const linkedin = document.getElementById('edit-linkedin').value.trim();
            const facebook = document.getElementById('edit-facebook').value.trim();
            const instagram = document.getElementById('edit-instagram').value.trim();
            const other_handles = document.getElementById('edit-other-handles').value.trim();
            const role = document.getElementById('edit-role').value;
            const timezone = document.getElementById('edit-timezone').value.trim();
            const password = document.getElementById('edit-password').value;
            const errEl = document.getElementById('edit-user-error');
            errEl.style.display = 'none';
            const payload = {display_name, job_title, email, phone, address, github, linkedin, facebook, instagram, other_handles, role, timezone};
            if (password) payload.password = password;
            try {
                const res = await apiFetch(apiUrl(`/api/users/${uid}`), {
                    method: 'PATCH',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload),
                });
                const data = await res.json();
                if (!res.ok) { errEl.textContent = data.error || 'Failed to save'; errEl.style.display = 'block'; return; }
                closeEditUserModal();
                loadUsers();
            } catch (e) { errEl.textContent = e.message; errEl.style.display = 'block'; }
        }

        async function deactivateUser(uid) {
            if (!confirm('Deactivate this user?')) return;
            await apiFetch(apiUrl(`/api/users/${uid}`), {method: 'DELETE'});
            loadUsers();
        }

        async function activateUser(uid) {
            await apiFetch(apiUrl(`/api/users/${uid}`), {
                method: 'PATCH',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({is_active: 1}),
            });
            loadUsers();
        }

