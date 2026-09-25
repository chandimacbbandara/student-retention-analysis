import '../App.css';

const MetricsCard = ({ accuracy, f1, roc }) => (
  <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem', flexWrap: 'wrap' }}>
    <div style={{ background: 'rgba(255,255,255,0.05)', padding: '0.8rem', borderRadius: '8px', flex: '1 1 120px', border: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Accuracy</div>
      <div style={{ fontSize: '1.2rem', fontWeight: '600', color: '#10b981' }}>{accuracy}</div>
    </div>
    <div style={{ background: 'rgba(255,255,255,0.05)', padding: '0.8rem', borderRadius: '8px', flex: '1 1 120px', border: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>F1 Score</div>
      <div style={{ fontSize: '1.2rem', fontWeight: '600', color: '#a78bfa' }}>{f1}</div>
    </div>
    <div style={{ background: 'rgba(255,255,255,0.05)', padding: '0.8rem', borderRadius: '8px', flex: '1 1 120px', border: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>ROC-AUC</div>
      <div style={{ fontSize: '1.2rem', fontWeight: '600', color: '#38bdf8' }}>{roc}</div>
    </div>
  </div>
);

function HowItWorks() {
  return (
    <div className="app-container" style={{ alignItems: 'flex-start' }}>
      <header style={{ width: '100%', textAlign: 'left', marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>How It Works</h1>
        <p className="subtitle">Understanding the predictive models powering our AI</p>
      </header>

      <div className="glass-panel" style={{ marginBottom: '2rem' }}>
        <h2 style={{ color: '#c4b5fd', marginBottom: '1rem' }}>Final Elastic Net Model</h2>
        <p style={{ lineHeight: '1.6', marginBottom: '1rem', color: 'var(--text-muted)' }}>
          The Elastic Net model combines the strengths of both Ridge (L2) and Lasso (L1) regression. 
          It utilizes <strong>26 core features</strong> from the student dataset. 
          This model is highly effective when there are multiple correlated features, as it can group them together and select the most important ones while shrinking the coefficients of less important ones to prevent overfitting.
        </p>
        <MetricsCard accuracy="67.36%" f1="0.6878" roc="0.8354" />
      </div>

      <div className="glass-panel" style={{ marginBottom: '2rem' }}>
        <h2 style={{ color: '#c4b5fd', marginBottom: '1rem' }}>Final Lasso Model</h2>
        <p style={{ lineHeight: '1.6', marginBottom: '1rem', color: 'var(--text-muted)' }}>
          The Lasso (Least Absolute Shrinkage and Selection Operator) model also utilizes the <strong>26 core features</strong>.
          Its defining characteristic is the use of L1 regularization, which acts as an automatic feature selector by driving the weights of irrelevant features exactly to zero. 
          This results in a simpler, more interpretable model that focuses only on the strongest predictors of student retention.
        </p>
        <MetricsCard accuracy="68.26%" f1="0.6958" roc="0.8325" />
      </div>

      <div className="glass-panel" style={{ marginBottom: '2rem' }}>
        <h2 style={{ color: '#c4b5fd', marginBottom: '1rem' }}>Final XGBoost Model (Advanced)</h2>
        <p style={{ lineHeight: '1.6', marginBottom: '1.5rem', color: 'var(--text-muted)' }}>
          Our most advanced predictive engine is the XGBoost (Extreme Gradient Boosting) model. 
          Unlike the linear models above, XGBoost utilizes an expanded set of <strong>45 features</strong>, which includes engineered metrics such as <em>grade trends, risk factors, and moving averages</em>. 
          As a tree-based ensemble method, it excels at capturing complex, non-linear relationships and interactions between different student attributes that simpler models might miss.
        </p>
        <MetricsCard accuracy="70.30%" f1="0.7040" roc="0.8307" />
        
        <h3 style={{ color: '#f8fafc', margin: '2rem 0 1rem', fontSize: '1.2rem' }}>Simulated ARIMA Forecast</h3>
        <p style={{ lineHeight: '1.6', marginBottom: '1.5rem', color: 'var(--text-muted)' }}>
          While our current dataset provides a snapshot of student demographics and academic performance, forecasting models like ARIMA and advanced time-series analysis allow us to predict future retention trends.
          Below is a simulated forecast demonstrating how dropout rates can be projected into the future based on historical patterns.
        </p>
        
        <div style={{ textAlign: 'center', marginTop: '2rem', background: '#0f172a', padding: '1rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)' }}>
          <img 
            src="/forecast.png" 
            alt="Simulated ARIMA Forecast" 
            style={{ maxWidth: '100%', height: 'auto', borderRadius: '8px' }} 
          />
          <p style={{ marginTop: '1rem', fontSize: '0.9rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
            Figure 1: Hypothetical projection of student dropout rates using simulated random-walk historical data.
          </p>
        </div>
      </div>
    </div>
  );
}

export default HowItWorks;
