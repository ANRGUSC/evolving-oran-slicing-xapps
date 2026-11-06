def controller(history, state, sla, predictors):
    """
    Improved controller focusing on CQI trend and dynamic deep-null handling.
    Aims to balance meeting SLA targets, maximizing best-effort throughput,
    and minimizing churn, especially during channel instability.
    """

    # Initialize state on first call
    if state.get('initialized') is None:
        state.set('prev_share', sla.priority.min_share)
        state.set('cqi_history', [])
        state.set('deep_null_state', 'normal')  # States: normal, detecting, deep_null
        state.set('deep_null_counter', 0)
        state.set('target_mbps_adjusted', sla.priority.target_mbps)
        state.set('initialized', True)

    new_share = sla.priority.min_share
    target_mbps = sla.priority.target_mbps
    pool_size = sla.pool_max
    prev_share = state.get('prev_share', sla.priority.min_share)
    cqi_history = state.get('cqi_history', [])

    priority_sample = history.slices[0].last()

    if priority_sample is not None:
        cqi = priority_sample.cqi
        cqi_history.append(cqi)

        # Keep history length manageable
        if len(cqi_history) > 50:
            cqi_history.pop(0)
        state.set('cqi_history', cqi_history)

        deep_null_state = state.get('deep_null_state', 'normal')
        deep_null_counter = state.get('deep_null_counter', 0)
        target_mbps_adjusted = state.get('target_mbps_adjusted', target_mbps)

        # --- Deep Null Detection and Handling ---
        if len(cqi_history) >= 5:
            recent_cqi_avg = sum(cqi_history[-5:]) / 5.0

            if deep_null_state == 'normal':
                if recent_cqi_avg < 3.5: # Threshold for starting detection
                    state.set('deep_null_state', 'detecting')
                    state.set('deep_null_counter', 1)
                else:
                    # In normal state, if CQI is good, try to adjust target slightly higher for buffer
                    if recent_cqi_avg > 10:
                        state.set('target_mbps_adjusted', target_mbps * 1.05)
                    else:
                        state.set('target_mbps_adjusted', target_mbps)

            elif deep_null_state == 'detecting':
                if recent_cqi_avg < 3.0: # Confirming deep null
                    deep_null_counter += 1
                    if deep_null_counter >= 4: # More samples to confirm
                        state.set('deep_null_state', 'deep_null')
                        # In deep null, target is effectively unachievable, use minimum
                        target_mbps_adjusted = target_mbps * 0.5 # Very low effective target
                else: # Recovery detected during detection phase
                    state.set('deep_null_state', 'normal')
                    state.set('deep_null_counter', 0)
                    state.set('target_mbps_adjusted', target_mbps) # Reset adjusted target

            elif deep_null_state == 'deep_null':
                if recent_cqi_avg > 5.0: # Threshold for exiting deep null
                    deep_null_counter -= 1
                    if deep_null_counter <= 0: # Sufficient recovery
                        state.set('deep_null_state', 'normal')
                        state.set('deep_null_counter', 0)
                        state.set('target_mbps_adjusted', target_mbps) # Reset adjusted target
                else:
                    # Maintain deep_null state, keep target low
                    state.set('target_mbps_adjusted', target_mbps * 0.5)


        state.set('deep_null_state', deep_null_state)
        state.set('deep_null_counter', deep_null_counter)
        state.set('target_mbps_adjusted', target_mbps_adjusted)

        # --- Share Calculation ---
        if deep_null_state == 'deep_null':
            # In deep null, aggressively minimize share to leave PRBs for BE.
            # We still need to meet min_share, but aim for it.
            new_share = sla.priority.min_share
        else:
            # CQI-based estimate of Mbps per PRB.
            # The formula mbps_per_prb = max(0.06, 0.10 * (cqi ** 1.15)) is a reasonable starting point.
            # Adjusting the exponent and coefficient can tune sensitivity.
            # Using a slightly higher base and a bit more aggressive scaling for good CQI.
            mbps_per_prb_estimate = max(0.05, 0.08 * (cqi ** 1.2))

            # Calculate needed PRBs based on adjusted target and current estimate
            # Add a small buffer, especially when CQI is trending down.
            if len(cqi_history) >= 10:
                cqi_change_short = cqi_history[-1] - cqi_history[-5]
                cqi_change_long = cqi_history[-5] - cqi_history[-10]
                
                trend_sensitivity = 1.0 # Default
                if cqi_change_short < -2.0 or cqi_change_long < -1.0: # Rapid decline
                    trend_sensitivity = 1.25
                elif cqi_change_short > 2.0: # Rapid improvement
                    trend_sensitivity = 0.95
                
            else:
                trend_sensitivity = 1.1 # Default for short history

            prbs_needed_estimate = (target_mbps_adjusted * trend_sensitivity) / mbps_per_prb_estimate if mbps_per_prb_estimate > 0 else pool_size
            
            # Convert PRBs to share percentage
            share_needed = (prbs_needed_estimate / pool_size) * 100.0
            share_needed = max(sla.priority.min_share, min(sla.priority.max_share, share_needed))

            # Blending logic for churn reduction, favoring stability when target is met or CQI is low.
            # If share_needed is significantly higher than prev_share, take a larger step.
            # If share_needed is lower or close, take smaller steps, especially if CQI is stable/good.
            
            # Calculate difference and determine blend factor
            share_diff = share_needed - prev_share
            
            if share_diff > 5: # Need significant increase
                blend_factor = 0.6
            elif share_diff < -5: # Need significant decrease
                blend_factor = 0.4
            else: # Small adjustment or decrease
                if cqi < 4: # Low CQI, be conservative
                    blend_factor = 0.2
                elif cqi > 12: # High CQI, can be more responsive
                    blend_factor = 0.5
                else: # Moderate CQI
                    blend_factor = 0.3

            # Apply blend, ensuring we don't overshoot drastically
            new_share = int((1.0 - blend_factor) * prev_share + blend_factor * share_needed)
            
            # Ensure new_share respects bounds and is an integer
            new_share = max(sla.priority.min_share, min(sla.priority.max_share, new_share))
            new_share = int(round(new_share))

    # Always update the previous share for the next iteration
    state.set('prev_share', new_share)

    return new_share
