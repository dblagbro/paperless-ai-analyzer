# Web UI Setup Guide

The Paperless AI Analyzer includes a web-based dashboard for monitoring and management.

## Features

- **Real-time Monitoring**: Live updates of analysis activity
- **Recent Analyses**: View last 50 document analyses with results
- **Risk Scores**: See forensic risk assessments
- **Anomaly Details**: Browse detected anomalies by type
- **Profile Management**: View active profiles and staging suggestions
- **Statistics**: Total documents analyzed, anomalies found, high-risk count
- **Manual Triggers**: Analyze specific documents on demand (coming soon)

## Quick Access

### Direct Access (Development)

Access directly via port:
```
http://your-server:8051
```

### Via NGINX Reverse Proxy (Production)

Access via your domain:
```
https://www.voipguru.org/paperless-ai-analyzer
```

## NGINX Configuration

### Step 1: Add Configuration

Add this location block to your nginx configuration file:

```nginx
location /paperless-ai-analyzer/ {
    proxy_pass http://paperless-ai-analyzer:8051/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_buffering off;
}
```

The configuration is provided in `nginx-config.conf` for easy copy-paste.

### Step 2: Find Your NGINX Config

Your nginx config is likely at one of these locations:

```bash
# Check current nginx config
docker compose exec nginx cat /etc/nginx/conf.d/default.conf

# Or check the main config
docker compose exec nginx cat /etc/nginx/nginx.conf
```

Based on your docker-compose.yml, the config directory is:
```
./config/nginx/conf.d/
```

### Step 3: Add the Location Block

Create or edit a config file:

```bash
# Create new config file for the analyzer
cat > config/nginx/conf.d/paperless-ai-analyzer.conf << 'EOF'
location /paperless-ai-analyzer/ {
    proxy_pass http://paperless-ai-analyzer:8051/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_buffering off;
}
EOF
```

### Step 4: Reload NGINX

```bash
# Test configuration first
docker compose exec nginx nginx -t

# If OK, reload
docker compose exec nginx nginx -s reload
```

### Step 5: Access the UI

Open your browser to:
```
https://www.voipguru.org/paperless-ai-analyzer
```

## Dashboard Overview

### Status Bar

Shows 6 key metrics:
- **Status**: Running / Stopped indicator
- **Documents Analyzed**: Total count since startup
- **Anomalies Detected**: Total anomalies found
- **High Risk Documents**: Documents with risk score ≥ 70%
- **Active Profiles**: Number of loaded document type profiles
- **Last Update**: Timestamp of most recent analysis

### Recent Analyses Section

Displays the last 20 document analyses with:
- Document title and ID
- Risk score percentage
- Matched profile (or "No profile" if none matched)
- Timestamp
- Anomaly tags (if any were detected)

**Color Coding:**
- Blue border: Normal document
- Red border: Anomalies detected
- Yellow border: High forensic risk

### Active Profiles Section

Shows all loaded document type profiles with:
- Profile name
- Version number

### Staging Profiles Section (when applicable)

Appears when documents don't match any active profile and the system generates suggestions.

Shows:
- Suggested profile filename
- Creation timestamp
- Instructions for reviewing and promoting

## Auto-Refresh

The dashboard automatically refreshes every 10 seconds to show real-time updates.

## API Endpoints

The web UI is powered by these API endpoints:

- `GET /api/status` - Current analyzer status and statistics
- `GET /api/recent` - Recent analysis results
- `GET /api/profiles` - Active and staging profiles
- `GET /api/staging/<filename>` - View staging profile content
- `POST /api/trigger` - Manually trigger document analysis (body: `{"doc_id": 146}`)
- `GET /api/logs` - Recent log entries
- `GET /health` - Health check endpoint

### Example API Usage

```bash
# Get current status
curl http://localhost:8051/api/status

# Get recent analyses
curl http://localhost:8051/api/recent

# Trigger analysis of document 146
curl -X POST -H "Content-Type: application/json" \
  -d '{"doc_id": 146}' \
  http://localhost:8051/api/trigger
```

## Disabling the Web UI

If you don't want the web UI, disable it in docker-compose.yml:

```yaml
environment:
  WEB_UI_ENABLED: "false"
```

And remove the port mapping:

```yaml
# ports:
#   - "8051:8051"
```

Restart the service:

```bash
docker compose up -d paperless-ai-analyzer
```

## Troubleshooting

### Can't Access UI

1. Check container is running:
   ```bash
   docker compose ps paperless-ai-analyzer
   ```

2. Check if port is accessible:
   ```bash
   curl http://localhost:8051/health
   ```

3. Check logs:
   ```bash
   docker compose logs paperless-ai-analyzer | grep "Starting web UI"
   ```

### NGINX 502 Bad Gateway

1. Verify analyzer container name:
   ```bash
   docker compose ps paperless-ai-analyzer
   ```

2. Check they're on the same network:
   ```bash
   docker network inspect docker_default
   ```

3. Test from nginx container:
   ```bash
   docker compose exec nginx curl http://paperless-ai-analyzer:8051/health
   ```

### No Data Showing

1. Wait for first analysis to complete (check logs)
2. Manually trigger analysis:
   ```bash
   docker exec paperless-ai-analyzer python -m analyzer.main --doc-id 146
   ```
3. Refresh the browser

### CSS/Styling Issues

The UI uses inline styles, so there are no external CSS dependencies. If styling looks broken:

1. Hard refresh the browser (Ctrl+F5 or Cmd+Shift+R)
2. Check browser console for errors
3. Verify Flask is serving the template correctly:
   ```bash
   docker compose logs paperless-ai-analyzer | grep "GET /"
   ```

## Security Considerations

### Authentication

The web UI currently has **no authentication**. It's designed to run on an internal network or behind a reverse proxy with authentication.

**Recommendations:**

1. **Internal Network Only**: Don't expose port 8051 externally
2. **NGINX Basic Auth**: Add authentication in nginx:
   ```nginx
   location /paperless-ai-analyzer/ {
       auth_basic "Restricted Access";
       auth_basic_user_file /etc/nginx/.htpasswd;
       proxy_pass http://paperless-ai-analyzer:8051/;
       # ... rest of config
   }
   ```

3. **VPN**: Access only via VPN

### API Token Exposure

The UI doesn't display API tokens, but they're accessible via the status API. If this is a concern, restrict access to the UI.

## Future Enhancements

Planned features:
- Manual document analysis trigger from UI
- Profile promotion workflow (move staging to active)
- Advanced filtering and search
- Export analysis results
- Webhook configuration
- Email notifications for high-risk documents

## Support

- Main docs: `README.md`
- Profile system: `profiles/README.md`
- Quick start: `QUICKSTART.md`
