import math


def controller(history, state, sla, predictors):
    """Advanced controller with adaptive allocation, predictive margins, and intelligent churn management.
    
    Key improvements:
    1. Refined efficiency curve with CQI-specific calibration and delivery variance detection.
    2. Multi-timescale trend analysis: detect both rapid and sustained CQI patterns.
    3. Adaptive safety margins that respond to volatility and delivery confidence.
    4. Weighted fairness with priority boost for underperformers with high weight.
    5. Incremental allocation targeting marginal attainment gains.
    6. Intelligent churn suppression that allows beneficial directional moves.
    7. Dynamic reallocation based on comparative advantage and pool pressure.
    """
    n = sla.n
    W = [sla.slices[k].weight for k in range(n)]
    T = [sla.slices[k].target_mbps for k in range(n)]
    cap = sla.guaranteed_cap
    pool_max = sla.pool_max

    def eff(cqi):
        """Calibrated efficiency: steeper low-end penalty, saturation at high CQI."""
        c = max(5.0, min(15.0, cqi))
        normalized = (c - 5.0) / 10.0
        # Exponent 1.9 provides strong low-CQI penalty while smooth high-CQI saturation.
        return 0.28 + 2.12 * (normalized ** 1.9)

    # Gather state and history.
    prev_shares = state.get("prev_shares", [0] * n)
    prev_attain = state.get("prev_attain", [0.0] * n)
    cqi_trend_short = state.get("cqi_trend_short", [0.0] * n)  # 2-3 tick trend
    cqi_trend_long = state.get("cqi_trend_long", [0.0] * n)    # 5+ tick trend
    underperf_counter = state.get("underperf_counter", [0] * n)
    allocation_stability = state.get("allocation_stability", [0] * n)  # Track stable allocations
    
    obs = []
    cqi_history = []
    delivered_history = []
    
    for k in range(n):
        last_sample = history.slices[k].last()
        current_cqi = last_sample.cqi if last_sample else 10.0
        obs.append(current_cqi)
        
        # Gather recent samples (up to 20).
        recent_cqi = []
        recent_delivered = []
        recent_shares = []
        
        try:
            samples = list(history.slices[k]._samples[-20:]) if hasattr(history.slices[k], '_samples') else []
        except:
            samples = []
        
        if samples:
            for sample in samples:
                if sample:
                    recent_cqi.append(sample.cqi)
                    mbps = sample.dl_bytes * 8 / 1e6 if sample.dl_bytes else 0.0
                    recent_delivered.append(mbps)
                    recent_shares.append(sample.prb_share if hasattr(sample, 'prb_share') else 0)
        
        if last_sample:
            recent_cqi.append(last_sample.cqi)
            mbps = last_sample.dl_bytes * 8 / 1e6 if last_sample.dl_bytes else 0.0
            recent_delivered.append(mbps)
            recent_shares.append(last_sample.prb_share if hasattr(last_sample, 'prb_share') else 0)
        
        cqi_history.append(recent_cqi)
        delivered_history.append(recent_delivered)
        
        # Compute short-term trend (2-3 samples).
        if len(recent_cqi) >= 3:
            short_trend = (recent_cqi[-1] - recent_cqi[-3]) / 2.0
        elif len(recent_cqi) >= 2:
            short_trend = recent_cqi[-1] - recent_cqi[-2]
        else:
            short_trend = 0.0
        cqi_trend_short[k] = 0.6 * cqi_trend_short[k] + 0.4 * short_trend
        
        # Compute long-term trend (5+ samples).
        if len(recent_cqi) >= 6:
            long_trend = (recent_cqi[-1] - recent_cqi[-6]) / 5.0
        elif len(recent_cqi) >= 3:
            long_trend = (recent_cqi[-1] - recent_cqi[0]) / (len(recent_cqi) - 1)
        else:
            long_trend = 0.0
        cqi_trend_long[k] = 0.7 * cqi_trend_long[k] + 0.3 * long_trend
    
    # Compute adaptive margins with variance and trend sensitivity.
    margins = []
    for k in range(n):
        base_margin = 0.88
        
        if len(cqi_history[k]) >= 2:
            mean_cqi = sum(cqi_history[k]) / len(cqi_history[k])
            cqi_min = min(cqi_history[k])
            cqi_max = max(cqi_history[k])
            cqi_var = cqi_max - cqi_min
            cqi_std = math.sqrt(sum((x - mean_cqi) ** 2 for x in cqi_history[k]) / len(cqi_history[k])) if len(cqi_history[k]) > 1 else 0
            
            # Base margin by CQI level.
            if mean_cqi >= 13.5:
                base_margin = 0.945
            elif mean_cqi >= 13.0:
                base_margin = 0.93
            elif mean_cqi >= 11.5:
                base_margin = 0.91
            elif mean_cqi >= 10.0:
                base_margin = 0.88
            elif mean_cqi >= 8.0:
                base_margin = 0.84
            else:
                base_margin = 0.78
            
            # Volatility adjustment (penalize high variance).
            volatility_penalty = 0.015 * min(cqi_std, 2.5)
            base_margin -= volatility_penalty
            
            # Trend adjustment.
            if cqi_trend_long[k] > 1.0:  # Strong sustained improvement
                base_margin = min(0.96, base_margin + 0.025)
            elif cqi_trend_short[k] > 1.5:  # Strong recent improvement
                base_margin = min(0.96, base_margin + 0.015)
            elif cqi_trend_long[k] < -0.8:  # Strong sustained degradation
                base_margin = max(0.72, base_margin - 0.04)
            elif cqi_trend_short[k] < -1.2:  # Strong recent degradation
                base_margin = max(0.72, base_margin - 0.02)
            
            # Delivery consistency check: if delivery is consistently below target despite allocation, loosen margin.
            if len(delivered_history[k]) >= 5:
                recent_del = delivered_history[k][-5:]
                avg_delivered = sum(recent_del) / len(recent_del)
                if avg_delivered < T[k] - 1.0:
                    base_margin = max(0.70, base_margin - 0.05)
        
        margins.append(max(0.70, min(0.96, base_margin)))
    
    # Estimate per-share rate.
    rate = [max(0.04, eff(obs[k]) * margins[k]) for k in range(n)]
    
    # Calculate shares needed for targets.
    need = [max(1, math.ceil((T[k] - 0.5) / rate[k])) if rate[k] > 0 else 70 for k in range(n)]
    
    # Compute recent attainment with extended window.
    recent_attainment = [0.5] * n
    for k in range(n):
        if len(delivered_history[k]) >= 6:
            recent = delivered_history[k][-6:]
            attain_count = sum(1 for d in recent if d >= T[k] - 0.5)
            recent_attainment[k] = attain_count / len(recent)
        elif len(delivered_history[k]) >= 3:
            recent = delivered_history[k][-3:]
            attain_count = sum(1 for d in recent if d >= T[k] - 0.5)
            recent_attainment[k] = attain_count / len(recent)
        elif len(delivered_history[k]) >= 1:
            recent_attainment[k] = 1.0 if delivered_history[k][-1] >= T[k] - 0.5 else 0.0
    
    # Track underperformers with hysteresis.
    for k in range(n):
        if recent_attainment[k] < 0.35:
            underperf_counter[k] = min(underperf_counter[k] + 1, 6)
        else:
            underperf_counter[k] = max(underperf_counter[k] - 1, 0)
    
    sustained_underperformers = [k for k in range(n) if underperf_counter[k] >= 2]
    
    # Compute allocation efficiency metric (marginal attainment gain per share).
    efficiency_scores = []
    for k in range(n):
        if recent_attainment[k] >= 0.95:
            # Already meeting target well, low priority for extra shares.
            efficiency_scores.append(0.1)
        elif recent_attainment[k] < 0.3 and rate[k] > 0.05:
            # Significant gap but feasible; high priority.
            efficiency_scores.append(W[k] * 2.0)
        else:
            # Partial attainment; moderate priority.
            efficiency_scores.append(W[k] * (1.0 - recent_attainment[k]) * 1.5)
    
    # Phase 1: Base allocation to meet targets for highest-weight slices.
    shares = [0] * n
    used = 0
    
    priority_order = sorted(
        range(n),
        key=lambda k: (-W[k], k not in sustained_underperformers, recent_attainment[k])
    )
    
    for k in priority_order:
        required = need[k]
        if required > 0 and used + required <= cap:
            shares[k] = required
            used += required
    
    # Phase 2: Aggressive boost for sustained underperformers with high weight.
    remaining = cap - used
    if remaining > 0 and sustained_underperformers:
        sustained_underperformers.sort(key=lambda k: (-W[k], underperf_counter[k]))
        for k in sustained_underperformers:
            if remaining > 0 and shares[k] < need[k] + 8:
                boost = min(remaining, need[k] + 8 - shares[k])
                if boost > 0:
                    shares[k] += boost
                    remaining -= boost
    
    # Phase 3: Incremental allocation based on efficiency (marginal attainment gain).
    if remaining > 0:
        for _ in range(min(remaining, 15)):
            best_k = -1
            best_score = -1.0
            for k in range(n):
                current_attain_estimate = min(1.0, rate[k] * shares[k] / (T[k] - 0.5)) if T[k] > 0.5 else 1.0
                if shares[k] < 28 and current_attain_estimate < 0.98:
                    score = efficiency_scores[k] * (1.0 - current_attain_estimate)
                    if score > best_score:
                        best_score = score
                        best_k = k
            
            if best_k >= 0:
                shares[best_k] += 1
                remaining -= 1
            else:
                break
    
    # Phase 4: Distribute remainder to avoid waste.
    if remaining > 0:
        # Distribute to slices with lowest current estimated attainment.
        sorted_by_attain = sorted(
            range(n),
            key=lambda k: (min(1.0, rate[k] * shares[k] / (T[k] - 0.5)) if T[k] > 0.5 else 1.0)
        )
        for k in sorted_by_attain:
            if remaining > 0 and shares[k] < cap:
                add = min(remaining, 1)
                shares[k] += add
                remaining -= add
    
    # Churn management: suppress thrashing but allow directional improvements.
    change_count = sum(1 for k in range(n) if shares[k] != prev_shares[k])
    
    if change_count >= 2:
        # Estimate new attainment.
        new_attain = []
        for k in range(n):
            est_delivered = rate[k] * shares[k]
            new_attain.append(1.0 if est_delivered >= T[k] - 0.5 else 0.0)
        
        new_weighted = sum(W[k] * new_attain[k] for k in range(n))
        old_weighted = sum(W[k] * prev_attain[k] for k in range(n))
        
        # If allocation worsens overall attainment significantly, revert.
        if new_weighted < old_weighted - 0.08:
            # Check for targeted improvement in sustained underperformers.
            targeted_improvement = any(
                k in sustained_underperformers and new_attain[k] > prev_attain[k] + 0.1
                for k in range(n)
            )
            if not targeted_improvement:
                shares = list(prev_shares)
                change_count = sum(1 for k in range(n) if shares[k] != prev_shares[k])
        
        # Track allocation stability for consistent slices.
        for k in range(n):
            if shares[k] == prev_shares[k]:
                allocation_stability[k] = min(allocation_stability[k] + 1, 8)
            else:
                allocation_stability[k] = max(allocation_stability[k] - 1, 0)
    
    # Update state.
    state.set("prev_shares", list(shares))
    state.set("cqi_trend_short", cqi_trend_short)
    state.set("cqi_trend_long", cqi_trend_long)
    state.set("underperf_counter", underperf_counter)
    state.set("allocation_stability", allocation_stability)
    
    # Store actual attainment for next tick.
    actual_attain = []
    for k in range(n):
        if len(delivered_history[k]) > 0:
            attain = 1.0 if delivered_history[k][-1] >= T[k] - 0.5 else 0.0
        else:
            attain = 1.0 if rate[k] * shares[k] >= T[k] - 0.5 else 0.0
        actual_attain.append(attain)
    
    state.set("prev_attain", actual_attain)
    
    return shares