import torch

# ============================
# PRINT TOGGLE
# ============================
DEBUG_PRINTS = False  # ← set to True to enable all prints


def debug_print(*args, **kwargs):
    if DEBUG_PRINTS:
        print(*args, **kwargs)


class ConditioningNoiseInjection:
    BOUNDARY_EPSILON = 1e-3
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 10, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "seed_from_js": ("INT", {"default": 0}),
                "batch_size_from_js": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "inject_noise"
    CATEGORY = "advanced/conditioning"

    @classmethod
    def IS_CHANGED(s, conditioning, threshold, strength, seed_from_js=0, batch_size_from_js=1, **kwargs):
        # We include batch_size in the hash to ensure updates when batch size changes
        return f"{seed_from_js}_{batch_size_from_js}_{threshold}_{strength}"

    def inject_noise(self, conditioning, threshold, strength, seed_from_js=0, batch_size_from_js=1, **kwargs):
        debug_print(f"\n[NoiseInjection] Base Seed: {seed_from_js}, Target Batch Size: {batch_size_from_js}")

        c_out = []
        
        # Round threshold for consistency
        threshold = round(threshold, 6)
        
        # Handle edge cases explicitly
        if threshold <= 0:
            # threshold = 0: noisy part should NEVER be active
            # clean part should be active for the entire range
            noisy_end = 0.0      # [0.0, 0.0) = empty range
            clean_start = 0.0   # [0.0, 1.0) = full range
            debug_print("EDGE CASE: threshold <= 0")
            debug_print("  Noisy part will be SILENCED (empty range)")
            debug_print("  Clean part will cover [0.0, 1.0)")
        elif threshold >= 1:
            # threshold = 1: noisy part should be active for the entire range
            # clean part should NEVER be active
            noisy_end = 1.0      # [0.0, 1.0) = full range
            clean_start = 1.0   # [1.0, 1.0) = empty range
            debug_print("EDGE CASE: threshold >= 1")
            debug_print("  Noisy part will cover [0.0, 1.0)")
            debug_print("  Clean part will be SILENCED (empty range)")
        else:
            # Normal case: add epsilon to include threshold point in noisy part
            noisy_end = threshold + self.BOUNDARY_EPSILON
            clean_start = threshold + self.BOUNDARY_EPSILON
            debug_print(f"NORMAL CASE: threshold = {threshold}")
            debug_print(f"  Noisy part: [0.0, {noisy_end})")
            debug_print(f"  Clean part: [{clean_start}, 1.0)")

        def get_time_intersection(params, limit_start, limit_end):
            old_start = params.get("start_percent", 0.0)
            old_end = params.get("end_percent", 1.0)
            
            debug_print(f"  get_time_intersection:")
            debug_print(f"    Input: [{old_start}, {old_end}]")
            debug_print(f"    Limits: [{limit_start}, {limit_end}]")

            new_start = max(old_start, limit_start)
            new_end = min(old_end, limit_end)
            
            debug_print(f"    Result: [{new_start}, {new_end}]")

            if new_start >= new_end:
                debug_print(f"    INVALID -> Silencing")
                return 1.0, 0.0
            
            return new_start, new_end

        # Standard ComfyUI CPU Generator for reproducibility
        g = torch.Generator(device="cpu")
        g.manual_seed(seed_from_js)

        for i, t in enumerate(conditioning):
            original_tensor = t[0] # Shape [Batch, Tokens, Channels] or [1, T, C]
            original_dict = t[1].copy()
            
            # 1. Handle Batch Expansion
            # If the input is Batch 1 (common for conditioning), but the workflow is Batch N,
            # we repeat the tensor to match.
            current_batch_count = original_tensor.shape[0]
            target_batch_count = max(current_batch_count, batch_size_from_js)
            
            processing_tensor = original_tensor
            if current_batch_count == 1 and target_batch_count > 1:
                # [1, T, C] -> [N, T, C]
                processing_tensor = original_tensor.repeat(target_batch_count, 1, 1)
            
            # 2. Generate Noise (Native ComfyUI Method)
            # We generate one large noise tensor for the whole batch at once.
            # This ensures Image 2 gets the "next" sequence of random numbers, consistent with KSampler.
            noise = torch.randn(
                processing_tensor.size(),
                generator=g,
                device="cpu"
            ).to(
                processing_tensor.device,
                dtype=processing_tensor.dtype
            )

            # --- VERIFICATION PRINTS ---
            debug_print(f"--- Conditioning Group {i} Noise Values ---")
            for b in range(target_batch_count):
                # Get the first few float values of the first token for this batch item
                # noise[b] is shape [Tokens, Channels]
                first_vals = noise[b, 0, :5].tolist()
                formatted_vals = [f"{x:+.4f}" for x in first_vals]
                debug_print(f"   > Batch Index {b}: {formatted_vals} ...")
            # ---------------------------

            # Apply noise
            noisy_tensor = processing_tensor + (noise * strength)

            # 3. Time Intersection & Output using the computed boundaries
            debug_print("-" * 40)
            debug_print(f"PROCESSING Conditioning Group {i}")
            debug_print("-" * 40)
            
            # Noisy Part (Start -> noisy_end)
            debug_print(f"Processing NOISY part (Range: 0.0 -> {noisy_end}):")
            s_val_noise, e_val_noise = get_time_intersection(original_dict, 0.0, noisy_end)
            n_noisy = [noisy_tensor, original_dict.copy()]
            n_noisy[1]["start_percent"] = s_val_noise
            n_noisy[1]["end_percent"] = e_val_noise
            debug_print(f"  FINAL: [{s_val_noise}, {e_val_noise}]")
            c_out.append(n_noisy)

            # Clean Part (clean_start -> End)
            debug_print(f"Processing CLEAN part (Range: {clean_start} -> 1.0):")
            s_val_clean, e_val_clean = get_time_intersection(original_dict, clean_start, 1.0)
            n_clean = [processing_tensor, original_dict.copy()]
            n_clean[1]["start_percent"] = s_val_clean
            n_clean[1]["end_percent"] = e_val_clean
            debug_print(f"  FINAL: [{s_val_clean}, {e_val_clean}]")
            c_out.append(n_clean)

        # Summary
        debug_print("=" * 60)
        debug_print("FINAL OUTPUT SUMMARY")
        debug_print(f"Total conditionings: {len(c_out)}")
        for i, c in enumerate(c_out):
            start = c[1].get("start_percent", "N/A")
            end = c[1].get("end_percent", "N/A")
            silenced = "(SILENCED)" if start == 1.0 and end == 0.0 else ""
            debug_print(f"  c_out[{i}]: [{start}, {end}] {silenced}")
        debug_print("=" * 60)

        return (c_out, )


NODE_CLASS_MAPPINGS = {
    "ConditioningNoiseInjection": ConditioningNoiseInjection
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningNoiseInjection": "Conditioning Noise Injection"
}

WEB_DIRECTORY = "./js"
