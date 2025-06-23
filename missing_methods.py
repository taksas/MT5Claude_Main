# Missing methods to add to trading_strategy.py

def _analyze_fractal_patterns(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """Analyze fractal patterns in price data"""
    try:
        fractal_data = {
            'dimension': 1.5,  # Default fractal dimension
            'self_similarity': 0,
            'fractal_levels': [],
            'golden_ratio_present': False,
            'fibonacci_confluence': 0
        }
        
        if len(df) >= 100:
            prices = df['close'].values
            
            # Calculate fractal dimension using box-counting method
            dimension = self._calculate_fractal_dimension(prices)
            fractal_data['dimension'] = dimension
            
            # Detect self-similar patterns at multiple scales
            scales = [5, 13, 21, 34, 55, 89]  # Fibonacci numbers
            similarities = []
            
            for scale in scales:
                if len(prices) >= scale * 2:
                    pattern1 = prices[-scale:]
                    pattern2 = prices[-scale*2:-scale]
                    
                    # Normalize patterns
                    pattern1_norm = (pattern1 - pattern1.min()) / (pattern1.max() - pattern1.min() + 1e-10)
                    pattern2_norm = (pattern2 - pattern2.min()) / (pattern2.max() - pattern2.min() + 1e-10)
                    
                    # Calculate similarity
                    similarity = np.corrcoef(pattern1_norm, pattern2_norm)[0, 1]
                    if not np.isnan(similarity):
                        similarities.append(similarity)
                        
                        if similarity > 0.8:
                            fractal_data['fractal_levels'].append({
                                'scale': scale,
                                'similarity': similarity
                            })
            
            if similarities:
                fractal_data['self_similarity'] = np.mean(similarities)
            
            # Check for golden ratio
            phi = 1.618033988749895
            recent_high = prices[-20:].max()
            recent_low = prices[-20:].min()
            range_size = recent_high - recent_low
            
            if range_size > 0:
                current_ratio = (current_price - recent_low) / range_size
                golden_ratios = [0.236, 0.382, 0.618, 0.786]
                
                for ratio in golden_ratios:
                    if abs(current_ratio - ratio) < 0.05:
                        fractal_data['golden_ratio_present'] = True
                        fractal_data['fibonacci_confluence'] += 1
        
        return fractal_data
    except Exception as e:
        logger.error(f"Error analyzing fractal patterns: {e}")
        return {'dimension': 1.5, 'self_similarity': 0, 'fractal_levels': [], 
               'golden_ratio_present': False, 'fibonacci_confluence': 0}

def _apply_chaos_theory(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """Apply chaos theory to predict market behavior"""
    try:
        chaos_data = {
            'chaos_level': 0,
            'attractor_type': 'point',
            'butterfly_effect': False,
            'prediction_horizon': 0,
            'lyapunov_exponent': 0
        }
        
        if len(df) >= 50:
            returns = df['close'].pct_change().dropna()
            
            # Calculate Lyapunov exponent
            lyapunov = self._calculate_lyapunov_exponent(returns.iloc[-50:])
            chaos_data['lyapunov_exponent'] = lyapunov
            
            # Determine chaos level
            if lyapunov > 0.1:
                chaos_data['chaos_level'] = min(1.0, lyapunov)
                chaos_data['attractor_type'] = 'strange'
                chaos_data['butterfly_effect'] = True
                chaos_data['prediction_horizon'] = int(1 / (lyapunov + 0.01))
            elif lyapunov > 0:
                chaos_data['chaos_level'] = lyapunov * 10
                chaos_data['attractor_type'] = 'limit_cycle'
                chaos_data['prediction_horizon'] = int(5 / (lyapunov + 0.01))
            else:
                chaos_data['attractor_type'] = 'point'
                chaos_data['prediction_horizon'] = 20
            
            # Detect strange attractors
            if len(returns) >= 100:
                # Phase space reconstruction
                embedded = self._embed_time_series(returns.iloc[-100:], 3, 1)
                if embedded is not None and len(embedded) > 0:
                    # Check for attractor patterns
                    variance = np.var(embedded, axis=0)
                    if np.any(variance > 0.01):
                        chaos_data['attractor_type'] = 'strange'
        
        return chaos_data
    except Exception as e:
        logger.error(f"Error applying chaos theory: {e}")
        return {'chaos_level': 0, 'attractor_type': 'point', 'butterfly_effect': False,
               'prediction_horizon': 0, 'lyapunov_exponent': 0}

def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
    """Calculate Hurst exponent for trend persistence"""
    try:
        n = len(returns)
        if n < 20:
            return 0.5
        
        # Convert to numpy array
        ts = returns.values
        
        # Calculate cumulative sum
        cumsum = np.cumsum(ts - np.mean(ts))
        
        # Calculate R/S for different lags
        lags = range(2, min(n//2, 20))
        rs_values = []
        
        for lag in lags:
            # Divide into chunks
            chunks = [cumsum[i:i+lag] for i in range(0, n-lag+1, lag)]
            
            rs_chunk = []
            for chunk in chunks:
                if len(chunk) == lag:
                    # Range and standard deviation
                    R = chunk.max() - chunk.min()
                    S = np.std(ts[len(cumsum)-len(chunk):len(cumsum)], ddof=1)
                    
                    if S > 0:
                        rs_chunk.append(R / S)
            
            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))
        
        if len(rs_values) > 3:
            # Log-log regression
            log_lags = np.log(list(lags)[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Fit line
            hurst = np.polyfit(log_lags, log_rs, 1)[0]
            return max(0, min(1, hurst))
        
        return 0.5
    except:
        return 0.5

def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
    """Calculate fractal dimension of price series"""
    try:
        n = len(prices)
        if n < 10:
            return 1.5
        
        # Normalize prices
        prices_norm = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)
        
        # Box counting method
        scales = np.logspace(0.01, 0.5, num=10, base=n)
        counts = []
        
        for scale in scales:
            scale_int = max(2, int(scale))
            
            # Grid boxes
            boxes = set()
            for i in range(0, n - scale_int):
                # Time box
                time_box = i // scale_int
                # Price box
                price_box = int(prices_norm[i] * scale_int)
                boxes.add((time_box, price_box))
            
            counts.append(len(boxes))
        
        # Calculate dimension
        if len(counts) > 3:
            coeffs = np.polyfit(np.log(1/scales[:len(counts)]), np.log(counts), 1)
            return abs(coeffs[0])
        
        return 1.5
    except:
        return 1.5

def _calculate_lyapunov_exponent(self, returns: pd.Series) -> float:
    """Calculate Lyapunov exponent for chaos detection"""
    try:
        n = len(returns)
        if n < 20:
            return 0
        
        # Embedding dimension
        m = 3
        # Time delay
        tau = 1
        
        # Create embedded time series
        embedded = self._embed_time_series(returns, m, tau)
        if embedded is None or len(embedded) < 10:
            return 0
        
        # Find nearest neighbors and track divergence
        lyapunov_sum = 0
        count = 0
        
        for i in range(len(embedded) - 1):
            # Find nearest neighbor
            distances = np.array([np.linalg.norm(embedded[i] - embedded[j]) 
                                 for j in range(len(embedded)) if j != i])
            
            if len(distances) > 0:
                nn_idx = np.argmin(distances)
                if nn_idx >= i:
                    nn_idx += 1
                
                # Track divergence
                if nn_idx < len(embedded) - 1:
                    initial_distance = distances[nn_idx - (1 if nn_idx > i else 0)]
                    if initial_distance > 1e-10:
                        final_distance = np.linalg.norm(embedded[i+1] - embedded[nn_idx+1])
                        if final_distance > 0:
                            lyapunov_sum += np.log(final_distance / initial_distance)
                            count += 1
        
        if count > 0:
            return lyapunov_sum / count
        
        return 0
    except:
        return 0

def _embed_time_series(self, series: pd.Series, m: int, tau: int) -> np.ndarray:
    """Embed time series in phase space"""
    try:
        n = len(series)
        if n < m * tau:
            return None
        
        embedded = np.zeros((n - (m-1)*tau, m))
        for i in range(m):
            embedded[:, i] = series.iloc[i*tau:n-(m-1-i)*tau].values
        
        return embedded
    except:
        return None