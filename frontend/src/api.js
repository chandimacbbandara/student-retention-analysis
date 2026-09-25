const API_URL = 'http://localhost:8000';

export const fetchModels = async () => {
    try {
        const response = await fetch(`${API_URL}/models`);
        if (!response.ok) throw new Error('Network response was not ok');
        return await response.json();
    } catch (error) {
        console.error('Error fetching models:', error);
        throw error;
    }
};

export const predictRetention = async (modelId, features) => {
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_id: modelId,
                features: features
            })
        });
        if (!response.ok) throw new Error('Prediction failed');
        return await response.json();
    } catch (error) {
        console.error('Error during prediction:', error);
        throw error;
    }
};
