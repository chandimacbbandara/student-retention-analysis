import { useState, useEffect } from 'react';
import '../App.css';
import { fetchModels, predictRetention } from '../api';

function Home() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [features, setFeatures] = useState([]);
  const [formData, setFormData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Fetch models on component mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        const data = await fetchModels();
        // Remove 'Best Model' from display if requested by the user
        const filteredModels = data.models.filter(m => m.name !== 'Best Model');
        setModels(filteredModels);
        if (filteredModels.length > 0) {
          handleModelSelect(filteredModels[0].id, filteredModels);
        }
      } catch (err) {
        setError('Failed to connect to the backend API. Ensure it is running on port 8000.');
      }
    };
    loadModels();
  }, []);

  const handleModelSelect = (modelId, modelsList = models) => {
    setSelectedModel(modelId);
    const model = modelsList.find(m => m.id === modelId);
    if (model) {
      setFeatures(model.features);
      const initialData = {};
      model.features.forEach(feature => {
        initialData[feature] = '';
      });
      setFormData(initialData);
      setPrediction(null);
    }
  };

  const handleRandomize = (e) => {
    e.preventDefault();
    const randomData = {};
    features.forEach(feature => {
      const fName = feature.toLowerCase();
      let val = 0;
      
      if (fName.includes('age')) {
        val = Math.floor(Math.random() * 30) + 18; // 18-47
      } else if (fName.includes('grade')) {
        val = Number((Math.random() * 10 + 10).toFixed(1)); // 10.0-20.0
      } else if (fName.includes('rate') || fName.includes('gdp') || fName.includes('inflation')) {
        val = Number((Math.random() * 10 - 2).toFixed(1)); // -2.0 to 8.0
      } else if (fName.includes('evaluations') || fName.includes('enrolled') || fName.includes('approved') || fName.includes('credited')) {
        val = Math.floor(Math.random() * 15); // 0-14
      } else if (fName.includes('trend') || fName.includes('risk') || fName.includes('mean')) {
        val = Number((Math.random() * 5).toFixed(2));
      } else if (fName.includes('course') || fName.includes('qualification') || fName.includes('occupation') || fName.includes('nationality')) {
        val = Math.floor(Math.random() * 20) + 1; // 1-20 for larger categorical
      } else {
        val = Math.floor(Math.random() * 2); // 0 or 1 for binary
      }
      randomData[feature] = val;
    });
    setFormData(randomData);
    setPrediction(null);
  };

  const handleInputChange = (e, feature) => {
    const value = e.target.value === '' ? '' : Number(e.target.value);
    setFormData(prev => ({
      ...prev,
      [feature]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    // Ensure all empty strings are converted to 0
    const submitData = { ...formData };
    Object.keys(submitData).forEach(key => {
      if (submitData[key] === '') submitData[key] = 0;
    });

    try {
      const result = await predictRetention(selectedModel, submitData);
      setPrediction(result);
    } catch (err) {
      setError('Prediction failed. Check console for details.');
    } finally {
      setLoading(false);
    }
  };

  const formatFeatureName = (name) => {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const getFeatureDescription = (name) => {
    const lower = name.toLowerCase();
    if (lower.includes('age')) return 'Years (e.g. 18, 19)';
    if (lower.includes('gender')) return '0 = Male, 1 = Female';
    if (lower.includes('debtor') || lower.includes('tuition') || lower.includes('scholarship') || lower.includes('displaced') || lower.includes('international') || lower.includes('special needs')) return '0 = No, 1 = Yes';
    if (lower.includes('grade')) return '0.0 to 20.0';
    if (lower.includes('rate') || lower.includes('gdp')) return 'Percentage (e.g. 10.8, -0.3)';
    if (lower.includes('curricular')) return 'Count (e.g. 0, 5, 6)';
    if (lower.includes('course') || lower.includes('mode') || lower.includes('order') || lower.includes('qualification') || lower.includes('occupation') || lower.includes('status') || lower.includes('nationality')) return 'Categorical Code (1, 2, 3...)';
    if (lower.includes('attendance')) return '0 = Evening, 1 = Daytime';
    if (lower.includes('trend') || lower.includes('risk') || lower.includes('mean')) return 'Engineered Metric';
    return 'Numeric value';
  };

  const getOptionsForFeature = (name) => {
    const lower = name.toLowerCase();
    if (lower.includes('gender')) return [{v: 0, l: '0 - Male'}, {v: 1, l: '1 - Female'}];
    if (lower.includes('debtor') || lower.includes('tuition') || lower.includes('scholarship') || lower.includes('displaced') || lower.includes('international') || lower.includes('special needs')) return [{v: 0, l: '0 - No'}, {v: 1, l: '1 - Yes'}];
    if (lower.includes('attendance')) return [{v: 0, l: '0 - Evening'}, {v: 1, l: '1 - Daytime'}];
    if (lower.includes('marital')) return [
      {v: 1, l: '1 - Single'}, {v: 2, l: '2 - Married'}, {v: 3, l: '3 - Widower'}, 
      {v: 4, l: '4 - Divorced'}, {v: 5, l: '5 - Common-law'}, {v: 6, l: '6 - Legally separated'}
    ];
    if (lower.includes('course')) return [
      {v: 1, l: '1 - Biofuel Prod'}, {v: 2, l: '2 - Animation'}, {v: 3, l: '3 - Social Service (eve)'},
      {v: 4, l: '4 - Agronomy'}, {v: 5, l: '5 - Comm Design'}, {v: 6, l: '6 - Vet Nursing'},
      {v: 7, l: '7 - Informatics Eng'}, {v: 8, l: '8 - Equiniculture'}, {v: 9, l: '9 - Management'},
      {v: 10, l: '10 - Social Service'}, {v: 11, l: '11 - Tourism'}, {v: 12, l: '12 - Nursing'},
      {v: 13, l: '13 - Oral Hygiene'}, {v: 14, l: '14 - Advertising'}, {v: 15, l: '15 - Journalism'},
      {v: 16, l: '16 - Basic Education'}, {v: 17, l: '17 - Management (eve)'}
    ];
    if (lower.includes('application mode')) return [
      {v: 1, l: '1 - 1st phase (general)'}, {v: 2, l: '2 - Ordinance 612/93'}, {v: 3, l: '3 - 1st phase (Azores)'},
      {v: 4, l: '4 - Other higher courses'}, {v: 5, l: '5 - Ordinance 854-B/99'}, {v: 6, l: '6 - Int. student'},
      {v: 7, l: '7 - 1st phase (Madeira)'}, {v: 8, l: '8 - 2nd phase (general)'}, {v: 9, l: '9 - 3rd phase (general)'},
      {v: 10, l: '10 - Ord 533-A/99(b2)'}, {v: 11, l: '11 - Ord 533-A/99(b3)'}, {v: 12, l: '12 - Over 23 years'},
      {v: 13, l: '13 - Transfer'}, {v: 14, l: '14 - Change in course'}, {v: 15, l: '15 - Tech spec diploma'},
      {v: 16, l: '16 - Change in inst/course'}, {v: 17, l: '17 - Short cycle diploma'}, {v: 18, l: '18 - Change (Int)'}
    ];
    return null;
  };

  return (
    <div className="app-container">
      <header>
        <h1>Student Retention AI</h1>
        <p className="subtitle">Advanced predictive analytics for student success</p>
      </header>

      {error && <div className="glass-panel" style={{ color: 'var(--danger)', marginBottom: '2rem', textAlign: 'center' }}>{error}</div>}

      <div className="glass-panel">
        <div className="model-selector" style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div style={{ flex: 1 }}>
            <label htmlFor="model-select">Select Prediction Model</label>
            <select 
              id="model-select"
              value={selectedModel}
              onChange={(e) => handleModelSelect(e.target.value)}
            >
              {models.map(model => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.features.length} features)
                </option>
              ))}
            </select>
          </div>
          <button 
            onClick={handleRandomize}
            className="btn-secondary" 
            style={{ marginBottom: '2px', padding: '1rem 1.5rem', whiteSpace: 'nowrap' }}
            type="button"
          >
            🎲 Random Fill
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            {features.map((feature, idx) => (
              <div className="input-group" key={idx}>
                <label title={feature}>
                  {formatFeatureName(feature)}
                  <span style={{ display: 'block', fontSize: '0.75rem', color: '#a78bfa', marginTop: '4px', fontWeight: '400', opacity: 0.8 }}>
                    {getFeatureDescription(feature)}
                  </span>
                </label>
                {getOptionsForFeature(feature) ? (
                  <select
                    value={formData[feature] === '' ? '' : formData[feature]}
                    onChange={(e) => handleInputChange(e, feature)}
                  >
                    <option value="">-- Select --</option>
                    {getOptionsForFeature(feature).map(opt => (
                      <option key={opt.v} value={opt.v}>{opt.l}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    step="any"
                    value={formData[feature] === '' ? '' : formData[feature]}
                    onChange={(e) => handleInputChange(e, feature)}
                    placeholder="0"
                  />
                )}
              </div>
            ))}
          </div>

          <button type="submit" className="btn-primary" disabled={loading || !selectedModel}>
            {loading ? <span className="loader"></span> : 'Predict Student Outcome'}
          </button>
        </form>
      </div>

      {prediction && (
        <div className="glass-panel results-container">
          <h2 style={{ marginBottom: '1.5rem', color: 'var(--text-main)' }}>Prediction Result</h2>
          
          <div className={`prediction-badge status-${prediction.prediction}`}>
            {prediction.prediction}
          </div>
          
          <div className="probabilities">
            {Object.entries(prediction.probabilities).map(([className, prob]) => (
              <div className="prob-card" key={className}>
                <span className="prob-label">{className}</span>
                <span className="prob-value" style={{ 
                  color: className === prediction.prediction ? 'var(--primary)' : 'var(--text-main)'
                }}>
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default Home;
