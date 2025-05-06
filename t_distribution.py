import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, t
import time

# Set page config
st.set_page_config(page_title="Confidence Interval Simulator", layout="wide")
st.title("Confidence Interval Simulation")

st.markdown("""
This app demonstrates why using the Z-distribution with a sample standard deviation produces
confidence intervals with **lower than nominal coverage**. Both methods estimate σ from the sample, 
but only the t-distribution correctly accounts for this additional source of uncertainty.
""")

# Sidebar for parameter inputs
st.sidebar.header("Simulation Parameters")

# Population parameters (true values, unknown in real experiments)
population_mean = st.sidebar.slider("True Population Mean (μ)", 0, 100, 50)
population_std = st.sidebar.slider("True Population Standard Deviation (σ)", 1, 30, 15)

# Sample and simulation parameters
sample_size = st.sidebar.slider("Sample Size (n)", 5, 100, 15, 
                               help="Small sample sizes (< 30) show larger differences between methods")
confidence_level = st.sidebar.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01, 
                                   format="%.2f")
num_simulations = st.sidebar.slider("Number of Simulations", 100, 10000, 1000, 
                                  help="More simulations give more stable results")

# Simulation seed for reproducibility
seed = st.sidebar.number_input("Random Seed", 0, 99999, 42, 
                             help="Setting the same seed allows reproducible results")

# Method selection
method = st.sidebar.radio("Display Method", [
    "Z with Sample Std (Normal Distribution)",
    "T with Sample Std (t-Distribution)",
    "Compare Both Methods"
])

# Explanation of methods
with st.sidebar.expander("About the Methods"):
    st.markdown("""
    **Z with Sample Std:**
    - Uses the sample standard deviation (s) to estimate σ
    - Uses z-distribution quantiles (which assume known σ)
    - Formula: x̄ ± z(α/2) × (s/√n)
    - INCORRECT approach that ignores estimation uncertainty
    - Should produce LOWER than nominal coverage

    **T with Sample Std:**
    - Uses the sample standard deviation (s) to estimate σ
    - Uses t-distribution quantiles with (n-1) degrees of freedom
    - Formula: x̄ ± t(α/2, n-1) × (s/√n)
    - CORRECT approach that accounts for uncertainty in estimating σ
    - Should produce coverage close to the nominal confidence level
    
    Theory predicts that Z with sample std will have **undercoverage**,
    especially with smaller sample sizes.
    """)

# Run the simulation
if st.button("Run Simulation"):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Create progress bar
    progress_bar = st.progress(0)
    start_time = time.time()
    
    # Store results
    z_intervals = []
    t_intervals = []
    z_captured = 0
    t_captured = 0
    sample_stds = []  # Store sample standard deviations
    
    # For tracking coverage as simulations progress
    track_every = max(1, num_simulations // 100)  # Track approximately 100 points
    z_coverage_progression = []
    t_coverage_progression = []
    sim_points = []

    # Run simulations
    for i in range(num_simulations):
        # Update progress bar every 10 simulations
        if i % 10 == 0:
            progress_bar.progress((i + 1) / num_simulations)
            elapsed = time.time() - start_time
            if i > 0:
                est_total = elapsed * num_simulations / i
                est_remaining = est_total - elapsed
                st.sidebar.text(f"Est. time remaining: {est_remaining:.1f}s")
        
        # Generate random sample from normal distribution
        sample = np.random.normal(loc=population_mean, scale=population_std, size=sample_size)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)  # Sample standard deviation with Bessel's correction
        sample_stds.append(sample_std)
        
        # Z-score method with sample std
        z_score = norm.ppf(1 - (1 - confidence_level) / 2)
        z_margin_error = z_score * (sample_std / np.sqrt(sample_size))
        z_lower = sample_mean - z_margin_error
        z_upper = sample_mean + z_margin_error
        z_intervals.append((z_lower, sample_mean, z_upper))
        if z_lower <= population_mean <= z_upper:
            z_captured += 1
            
        # T-score method with sample std
        t_score = t.ppf(1 - (1 - confidence_level) / 2, sample_size - 1)
        t_margin_error = t_score * (sample_std / np.sqrt(sample_size))
        t_lower = sample_mean - t_margin_error
        t_upper = sample_mean + t_margin_error
        t_intervals.append((t_lower, sample_mean, t_upper))
        if t_lower <= population_mean <= t_upper:
            t_captured += 1
            
        # Track coverage progression
        if (i + 1) % track_every == 0:
            sim_points.append(i + 1)
            z_coverage_progression.append(z_captured / (i + 1) * 100)
            t_coverage_progression.append(t_captured / (i + 1) * 100)
    
    # Clear progress bar after completion
    progress_bar.empty()
    st.sidebar.text(f"Total time: {time.time() - start_time:.1f}s")
    
    # Calculate capture percentages
    z_capture_percentage = z_captured / num_simulations * 100
    t_capture_percentage = t_captured / num_simulations * 100
    
    # Results summary
    st.header("Simulation Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sample Size", sample_size)
    with col2:
        st.metric("Confidence Level", f"{confidence_level*100:.1f}%")
    with col3:
        st.metric("Number of Simulations", num_simulations)
    
    # Display results with color coding
    st.subheader("Coverage Results")
    
    # Calculate expected variation
    expected_variation = 1.96 * np.sqrt((confidence_level * (1-confidence_level)) / num_simulations) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Z-Method Coverage", f"{z_capture_percentage:.2f}%", 
                 delta=f"{z_capture_percentage - confidence_level*100:.2f}%", 
                 delta_color="inverse")
    with col2:
        st.metric("T-Method Coverage", f"{t_capture_percentage:.2f}%", 
                 delta=f"{t_capture_percentage - confidence_level*100:.2f}%", 
                 delta_color="inverse")
    with col3:
        st.metric("Expected Variation", f"±{expected_variation:.2f}%", 
                 help="95% of simulations should be within this range of the nominal confidence level")
    
    # Calculate critical values
    z_crit = norm.ppf(1 - (1 - confidence_level) / 2)
    t_crit = t.ppf(1 - (1 - confidence_level) / 2, sample_size - 1)
    crit_diff_pct = ((t_crit / z_crit) - 1) * 100
    
    # Calculate average interval widths
    z_widths = [upper - lower for lower, _, upper in z_intervals]
    t_widths = [upper - lower for lower, _, upper in t_intervals]
    avg_z_width = np.mean(z_widths)
    avg_t_width = np.mean(t_widths)
    width_diff_pct = ((avg_t_width / avg_z_width) - 1) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Critical Value (Z)", f"{z_crit:.4f}")
    with col2:
        st.metric("Critical Value (T)", f"{t_crit:.4f}")
    with col3:
        st.metric("T vs Z Difference", f"+{crit_diff_pct:.2f}%", 
                 help="How much larger the t-critical value is compared to z-critical")
        
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Z-Interval Width", f"{avg_z_width:.2f}")
    with col2:
        st.metric("Avg T-Interval Width", f"{avg_t_width:.2f}")
    with col3:
        st.metric("T vs Z Width Difference", f"+{width_diff_pct:.2f}%",
                help="How much wider t-intervals are compared to z-intervals")

    # Create a dataframe for easy comparison
    comparison_df = pd.DataFrame({
        'Method': ['Z with Sample Std', 'T with Sample Std'],
        'Critical Value': [f"{z_crit:.4f}", f"{t_crit:.4f}"],
        'Avg Interval Width': [f"{avg_z_width:.2f}", f"{avg_t_width:.2f}"],
        'Captures': [f"{z_captured}/{num_simulations}", f"{t_captured}/{num_simulations}"],
        'Coverage Rate': [f"{z_capture_percentage:.2f}%", f"{t_capture_percentage:.2f}%"],
        'Difference from Nominal': [f"{z_capture_percentage - confidence_level*100:.2f}%", 
                                   f"{t_capture_percentage - confidence_level*100:.2f}%"]
    })
    
    st.subheader("Method Comparison")
    st.dataframe(comparison_df, use_container_width=True)
        
    # T-test for significance
    diff_intervals = [(t_upper - t_lower) - (z_upper - z_lower) for (z_lower, _, z_upper), (t_lower, _, t_upper) 
                     in zip(z_intervals, t_intervals)]
    from scipy.stats import ttest_1samp
    t_stat, p_value = ttest_1samp(diff_intervals, 0)
    
    # Plot coverage progression
    st.subheader("Coverage Progression")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sim_points, z_coverage_progression, 'b-', label='Z-Method Coverage')
    ax.plot(sim_points, t_coverage_progression, 'r-', label='T-Method Coverage')
    ax.axhline(confidence_level*100, color='g', linestyle='--', label=f'Target ({confidence_level*100:.1f}%)')
    
    # Add bands for expected variation
    upper_bound = confidence_level*100 + expected_variation
    lower_bound = confidence_level*100 - expected_variation
    ax.fill_between(sim_points, lower_bound, upper_bound, color='green', alpha=0.1, 
                   label=f'Expected Variation (±{expected_variation:.2f}%)')
    
    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('Coverage Rate (%)')
    ax.set_title('Coverage Rate Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Create figure based on selected method for interval visualization
    if method == "Compare Both Methods":
        # Side by side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        
        # Plot Z intervals (first 50 or fewer)
        plot_count = min(50, num_simulations)
        for i in range(plot_count):
            low, mean, high = z_intervals[i]
            color = 'blue' if low <= population_mean <= high else 'red'
            ax1.plot([i, i], [low, high], color=color, alpha=0.7)
            ax1.plot(i, mean, 'ko', markersize=3)
        
        # Reference line for population mean
        ax1.axhline(population_mean, color='green', linestyle='-', label='True μ')
        ax1.set_title(f"Z-Method: {z_capture_percentage:.1f}% Coverage")
        ax1.set_xlabel("Simulation Index")
        ax1.set_ylabel("Interval")
        ax1.legend()
        
        # Plot T intervals (first 50 or fewer)
        for i in range(plot_count):
            low, mean, high = t_intervals[i]
            color = 'blue' if low <= population_mean <= high else 'red'
            ax2.plot([i, i], [low, high], color=color, alpha=0.7)
            ax2.plot(i, mean, 'ko', markersize=3)
        
        # Reference line for population mean
        ax2.axhline(population_mean, color='green', linestyle='-', label='True μ')
        ax2.set_title(f"T-Method: {t_capture_percentage:.1f}% Coverage")
        ax2.set_xlabel("Simulation Index")
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        # Single plot based on selected method
        intervals = z_intervals if method == "Z with Sample Std (Normal Distribution)" else t_intervals
        captured = z_captured if method == "Z with Sample Std (Normal Distribution)" else t_captured
        capture_percentage = z_capture_percentage if method == "Z with Sample Std (Normal Distribution)" else t_capture_percentage
        method_short = "Z" if method == "Z with Sample Std (Normal Distribution)" else "t"
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot intervals (first 50 or fewer)
        plot_count = min(50, num_simulations)
        for i in range(plot_count):
            low, mean, high = intervals[i]
            color = 'blue' if low <= population_mean <= high else 'red'
            ax.plot([i, i], [low, high], color=color, alpha=0.7)
            ax.plot(i, mean, 'ko', markersize=3)
        
        # Reference line for population mean
        ax.axhline(population_mean, color='green', linestyle='-', label='True Population Mean')
        
        # Add labels and title
        ax.set_xlabel("Simulation Index")
        ax.set_ylabel("Confidence Interval")
        ax.set_title(f"{confidence_level*100:.1f}% Confidence Intervals ({method_short}-distribution with sample std)\n"
                    f"Captured Mean in {captured} of {num_simulations} Simulations ({capture_percentage:.1f}%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Display plot in Streamlit
        st.pyplot(fig)
    
    # Analysis section
    st.header("Analysis")
    
    # Explanation based on results
    st.subheader("Interpretation")
    
    # Compare coverage rates from expected
    z_diff = abs(z_capture_percentage - (confidence_level * 100))
    t_diff = abs(t_capture_percentage - (confidence_level * 100))
    
    # Theoretical undercoverage
    theo_z_coverage = norm.cdf(z_crit * np.sqrt(sample_size)) - norm.cdf(-z_crit * np.sqrt(sample_size))
    theo_z_coverage_pct = theo_z_coverage * 100
    
    # Theoretical explanation
    st.markdown(f"""
    ### Statistical Theory Explanation
    
    **Why Z with Sample Std Should Show Undercoverage:**
    
    When using the sample standard deviation with z-critical values, we're treating 
    the sample std as if it were the true population std. This ignores the uncertainty
    in our estimate of σ, leading to intervals that are too narrow and capture the true
    mean less often than the nominal confidence level suggests.
    
    The t-distribution was specifically designed to handle this situation, by making the
    intervals wider to account for the extra uncertainty.
    
    **With your parameters:**
    - Sample size (n): {sample_size}
    - The t-critical value ({t_crit:.4f}) is {crit_diff_pct:.2f}% larger than the z-critical value ({z_crit:.4f})
    - This makes t-intervals {width_diff_pct:.2f}% wider than z-intervals on average
    
    **Observed vs. Expected Results:**
    - Z-method: {z_capture_percentage:.2f}% coverage (expected to be < {confidence_level*100:.1f}%)
    - T-method: {t_capture_percentage:.2f}% coverage (expected to be ≈ {confidence_level*100:.1f}%)
    """)
    
    if z_capture_percentage < confidence_level*100 - expected_variation:
        st.success(f"✓ The Z-method shows statistically significant undercoverage, as theory predicts.")
    elif z_capture_percentage < confidence_level*100:
        st.info(f"The Z-method shows undercoverage as expected, but the difference is not statistically significant with {num_simulations} simulations.")
    else:
        st.warning(f"The Z-method doesn't show undercoverage in this simulation. This can happen by chance, especially with fewer simulations. Try increasing the number of simulations or changing the random seed.")
    
    if abs(t_capture_percentage - confidence_level*100) <= expected_variation:
        st.success(f"✓ The T-method shows proper coverage within the expected variation.")
    else:
        st.info(f"The T-method coverage ({t_capture_percentage:.2f}%) differs from the nominal level ({confidence_level*100:.1f}%) by more than expected. This can happen by chance, especially with fewer simulations.")
    
    # Educational notes
    with st.expander("Educational Notes & Visualizations"):
        # Create columns
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown(f"""
            ### Key Concepts
            
            **Why t-distribution vs z-distribution?**
            
            When we estimate σ from the sample:
            
            1. **Additional uncertainty**: There's uncertainty in the sample mean AND in the sample std
            2. **Z-distribution ignores this**: It assumes we know σ exactly
            3. **T-distribution accounts for this**: It has heavier tails to account for this extra uncertainty
            4. **Sample size matters**: As n increases, the impact diminishes (t approaches z)
            
            **Mathematical perspective:**
            
            For a 95% confidence interval:
            - Z-interval: x̄ ± 1.96 × (s/√n)
            - T-interval: x̄ ± t<sub>0.975,n-1</sub> × (s/√n)
            
            For your sample size (n={sample_size}):
            - t<sub>0.975,{sample_size-1}</sub> = {t_crit:.4f}
            - This is {crit_diff_pct:.2f}% larger than 1.96
            
            **Common rules of thumb:**
            - For n < 30, always use t-distribution
            - For n ≥ 30, t and z give similar results (but t is still technically correct)
            """, unsafe_allow_html=True)
        
        with col2:
            # Show sample std vs population std
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Plot histogram of sample standard deviations
            ax.hist(sample_stds, bins=20, alpha=0.7, density=True)
            ax.axvline(population_std, color='red', linestyle='--', linewidth=2, label=f'True σ = {population_std}')
            ax.axvline(np.mean(sample_stds), color='blue', linestyle='-', linewidth=2, label=f'Mean s = {np.mean(sample_stds):.2f}')
            
            ax.set_title('Distribution of Sample Standard Deviations')
            ax.set_xlabel('Sample Standard Deviation')
            ax.set_ylabel('Density')
            ax.legend()
            st.pyplot(fig)
            
            st.markdown(f"""
            The histogram shows how sample standard deviations (s) vary around the true population standard deviation (σ).
            
            - Mean of sample stds: {np.mean(sample_stds):.2f}
            - True population std: {population_std}
            - Relative Bias: {(np.mean(sample_stds) - population_std) / population_std * 100:.2f}%
            """)
            
        # Comparison of distributions
        st.subheader("Z vs T Distribution Comparison")
        
        # Create x values for plotting distributions
        x = np.linspace(-4, 4, 1000)
        z_pdf = norm.pdf(x)
        
        # Create t distributions for different degrees of freedom
        dfs = [4, 9, 29, 99]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, z_pdf, 'k-', linewidth=2, label='Standard Normal (Z)')
        
        for df in dfs:
            t_pdf = t.pdf(x, df)
            ax.plot(x, t_pdf, '--', linewidth=1.5, label=f't with {df} df')
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.set_title('Comparison of Z and t Distributions')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Zoom in on the tails
        ax.set_xlim(1.5, 3.5)
        ax.set_ylim(0, 0.15)
        st.pyplot(fig)
        
        st.markdown("""
        This graph zooms in on the tails of the distributions where the critical values are located.
        Notice how the t-distribution has heavier tails (higher values) than the z-distribution, especially 
        with fewer degrees of freedom. This leads to larger critical values and wider confidence intervals.
        
        As the degrees of freedom increase (larger sample sizes), the t-distribution approaches the z-distribution.
        """)

# Run the app with: streamlit run app.py