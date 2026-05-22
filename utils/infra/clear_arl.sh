# Clean up stale ARL resources from the CURRENT experiment via Gateway API.
# NOTE: Pod termination is async on the K8s side. This script only triggers
# the cleanup; pods may still be terminating after the script exits.

EXPERIMENT_ID="${ARL_EXPERIMENT_ID:-default}"
GATEWAY_URL="${ARL_GATEWAY_URL:-http://localhost:8080}"

echo "Cleaning up ARL resources for experiment '$EXPERIMENT_ID' via gateway ($GATEWAY_URL)..."
resp=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE \
    "${GATEWAY_URL}/v1/managed/experiments/${EXPERIMENT_ID}")

if [ "$resp" = "200" ] || [ "$resp" = "204" ]; then
    echo "Gateway accepted cleanup request (HTTP $resp). Pods will terminate asynchronously."
elif [ "$resp" = "404" ]; then
    echo "No resources found for experiment '$EXPERIMENT_ID' (HTTP 404), nothing to clean."
else
    echo "Warning: Gateway cleanup returned HTTP $resp. Check gateway logs."
fi
