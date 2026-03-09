# Clean up stale ARL resources from the CURRENT experiment via Gateway API.
# No kubectl required — all cleanup goes through the ARL gateway.

EXPERIMENT_ID="${ARL_EXPERIMENT_ID:-default}"
GATEWAY_URL="${ARL_GATEWAY_URL:-http://localhost:8080}"

echo "Cleaning up ARL resources for experiment '$EXPERIMENT_ID' via gateway ($GATEWAY_URL)..."
resp=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE \
    "${GATEWAY_URL}/v1/managed/experiments/${EXPERIMENT_ID}")

if [ "$resp" = "200" ] || [ "$resp" = "204" ]; then
    echo "Gateway cleanup succeeded (HTTP $resp)."
elif [ "$resp" = "404" ]; then
    echo "No resources found for experiment '$EXPERIMENT_ID' (HTTP 404), nothing to clean."
else
    echo "Warning: Gateway cleanup returned HTTP $resp. Check gateway logs."
fi

# Poll the gateway until no sandboxes remain for this experiment.
echo "Waiting for experiment sandboxes to terminate..."
CLEANUP_TIMEOUT=120
CLEANUP_ELAPSED=0
while true; do
    sandbox_count=$(curl -sf "${GATEWAY_URL}/v1/managed/experiments/${EXPERIMENT_ID}/sandboxes" 2>/dev/null \
        | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")

    if [ "$sandbox_count" = "0" ]; then
        break
    fi

    if [ $CLEANUP_ELAPSED -ge $CLEANUP_TIMEOUT ]; then
        echo "Warning: $sandbox_count sandbox(es) still running after ${CLEANUP_TIMEOUT}s, proceeding anyway."
        break
    fi
    sleep 5
    CLEANUP_ELAPSED=$((CLEANUP_ELAPSED + 5))
done
echo "Cleanup done."
