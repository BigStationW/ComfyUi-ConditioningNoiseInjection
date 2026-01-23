import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/* ===========================
   DEBUG TOGGLE
   =========================== */
const DEBUG_LOGS = false; // ← set to true to enable all logs

function debugLog(...args) {
    if (DEBUG_LOGS) {
        console.log(...args);
    }
}

/**
 * Scans the workflow to find the active Sampler, its Seed, and the associated Batch Size.
 * Returns { seed: number, batchSize: number }
 */
function findWorkflowParams(app) {
    const graph = app.graph;
    let foundSeed = 0;
    let foundBatchSize = 1;
    let foundSampler = false;

    // Helper to extract batch size from a node if it looks like a Latent source
    function getBatchSizeFromNode(node) {
        // Standard EmptyLatentImage
        if (node.widgets) {
            const batchWidget = node.widgets.find(w => w.name === "batch_size");
            if (batchWidget) return batchWidget.value;
        }
        return 1;
    }

    // Helper to find input node
    function getSourceNodeByInput(targetNode, inputName) {
        const input = targetNode.inputs?.find(i => i.name === inputName);
        if (input && input.link) {
            const link = graph.links[input.link];
            if (link) {
                return graph._nodes_by_id[link.origin_id];
            }
        }
        return null;
    }

    // Strategy: Find the Sampler
    for (const node of graph._nodes) {
        const nodeType = node.type || node.constructor.type;
        
        // 1. Check SamplerCustomAdvanced (from your workflow)
        if (nodeType === "SamplerCustomAdvanced") {
            debugLog(`[NoiseInjection JS] Found SamplerCustomAdvanced: ${node.id}`);
            foundSampler = true;

            // Find Seed (from 'noise' input or widget)
            const noiseNode = getSourceNodeByInput(node, "noise");
            if (noiseNode) {
                const seedWidget = noiseNode.widgets?.find(w => w.name === "seed" || w.name === "noise_seed");
                if (seedWidget) foundSeed = seedWidget.value;
            }

            // Find Batch Size (from 'latent_image' input)
            const latentNode = getSourceNodeByInput(node, "latent_image");
            if (latentNode) {
                debugLog(`[NoiseInjection JS] Found latent source: ${latentNode.type}`);
                foundBatchSize = getBatchSizeFromNode(latentNode);
            }
            break; // Stop after finding the first active advanced sampler
        }

        // 2. Check Standard KSampler
        if (nodeType === "KSampler" || nodeType === "KSamplerAdvanced") {
             // Basic connectivity check
            const hasModel = node.inputs?.find(i => i.name === "model" && i.link);
            if (hasModel) {
                debugLog(`[NoiseInjection JS] Found KSampler: ${node.id}`);
                foundSampler = true;

                // Seed
                const seedWidget = node.widgets?.find(w => w.name === "seed" || w.name === "noise_seed");
                if (seedWidget) foundSeed = seedWidget.value;

                // Batch Size
                const latentNode = getSourceNodeByInput(node, "latent_image");
                if (latentNode) {
                    foundBatchSize = getBatchSizeFromNode(latentNode);
                }
                break;
            }
        }
    }

    if (!foundSampler) {
        debugLog("[NoiseInjection JS] No active sampler found. Using defaults.");
    }

    return { seed: foundSeed, batchSize: foundBatchSize };
}

app.registerExtension({
    name: "ConditioningNoiseInjection",

    async setup() {
        debugLog("[NoiseInjection JS] Extension loaded with Batch Support");

        const originalApiQueuePrompt = api.queuePrompt;

        api.queuePrompt = async function (number, { output, workflow }, options) {
            debugLog("[NoiseInjection JS] API queuePrompt intercepted");

            // Detect parameters from the current graph state
            const params = findWorkflowParams(app);
            debugLog(`[NoiseInjection JS] Detected - Seed: ${params.seed}, Batch: ${params.batchSize}`);

            for (const nodeId in output) {
                const nodeData = output[nodeId];
                if (nodeData.class_type === "ConditioningNoiseInjection") {
                    debugLog(`[NoiseInjection JS] Injecting params into Node ${nodeId}`);
                    
                    // Inject both seed and batch size
                    nodeData.inputs.seed_from_js = params.seed;
                    nodeData.inputs.batch_size_from_js = params.batchSize;
                }
            }

            return originalApiQueuePrompt.call(this, number, { output, workflow }, options);
        };
    }
});
