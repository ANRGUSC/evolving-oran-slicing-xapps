def controller(history, state, sla, predictors):
    """
    Improved PRB allocation controller with better deep-null detection and adaptive hysteresis.
    
    Key improvements:
    1. Track CQI history to distinguish transient dips from sustained deep nulls.
    2. Use exponential moving average (EMA) of CQI for smoother deep-null detection.
    3. Adaptive hysteresis: tighter (less churn) in stable periods, looser when recovering.
    4. Conservative share allocation: when CQI is uncertain, prefer lower share to avoid violations.
    5. Better calibration: account for the fact that reported CQI floor is ~5.0-5.5 in nulls.
    """
    
    # Get current sample for priority slice (index 0)
    priority_sample = history.slices[0].last()
    
    if priority_sample is None:
        return (sla.priority.min_share + sla.priority.max_share) // 2
    
    # Extract current metrics
    current_cqi = priority_sample.cqi if priority_sample.cqi is not None else 5.0
    current_mbps = history.slices[0].latest_mbps()
    current_share = priority_sample.prb_share if priority_sample.prb_share is not None else 50
    
    target_mbps = sla.priority.target_mbps
    pool_size = sla.pool_max
    
    # Initialize state tracking
    if state.get('cqi_history') is None:
        state.set('cqi_history', [])
        state.set('cqi_ema', current_cqi)
        state.set('null_detected_count', 0)
        state.set('prev_share', current_share)
    
    # Update CQI history (keep last 10 samples)
    cqi_history = state.get('cqi_history', [])
    cqi_history.append(current_cqi)
    if len(cqi_history) > 10:
        cqi_history.pop(0)
    state.set('cqi_history', cqi_history)
    
    # Compute exponential moving average of CQI (alpha=0.3 for responsiveness)
    cqi_ema = state.get('cqi_ema', current_cqi)
    cqi_ema = 0.3 * current_cqi + 0.7 * cqi_ema
    state.set('cqi_ema', cqi_ema)
    
    # Detect sustained deep null: CQI EMA < 6.5 AND recent actual CQI very low
    recent_low_cqi = sum(1 for c in cqi_history[-3:] if c < 6.0) >= 2
    is_deep_null = cqi_ema < 6.5 and recent_low_cqi
    
    # Count consecutive null ticks for stability assessment
    if is_deep_null:
        null_count = state.get('null_detected_count', 0) + 1
        state.set('null_detected_count', null_count)
    else:
        state.set('null_detected_count', 0)
    
    # Improved capacity estimation with better calibration
    # CQI 15 => ~2.3 Mbps/PRB, CQI 5 => ~0.15 Mbps/PRB
    # Use a slightly more aggressive slope to capture good-channel gains
    mbps_per_prb = 0.145 * cqi_ema + 0.08
    mbps_per_prb = max(0.08, mbps_per_prb)
    
    # Calculate PRB share needed
    prb_needed = target_mbps / mbps_per_prb
    share_needed = int((prb_needed / pool_size) * 100)
    share_needed = max(sla.priority.min_share, min(sla.priority.max_share, share_needed))
    
    # In deep null: use minimum share (accept that SLA cannot be met)
    if is_deep_null:
        new_share = sla.priority.min_share
    else:
        # Adaptive hysteresis based on stability
        # Stable good channel (CQI > 10 for 5+ ticks): tight hysteresis (1%)
        # Recovering channel (6-10): medium hysteresis (2%)
        # Entering null (5-6): loose hysteresis (3%) to avoid thrashing
        
        if cqi_ema > 10.0 and state.get('null_detected_count', 0) == 0:
            hysteresis = 1
        elif cqi_ema > 8.0:
            hysteresis = 2
        else:
            hysteresis = 3
        
        prev_share = state.get('prev_share', current_share)
        
        # Only change if the gain is significant relative to hysteresis
        if abs(share_needed - prev_share) >= hysteresis:
            new_share = share_needed
        else:
            new_share = prev_share
    
    # Final bounds check
    new_share = max(sla.priority.min_share, min(sla.priority.max_share, new_share))
    
    # Update persistent state
    state.set('prev_share', new_share)
    
    return new_share