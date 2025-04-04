document.getElementById('weather-form').addEventListener('submit', function (e) {
    e.preventDefault();

    const formData = new FormData(this);
    const params = new URLSearchParams(formData);

    fetch('/predict', {
        method: 'POST',
        body: params
    })
    .then(response => response.json())
    .then(data => {
        const result = data['Predicted Temperature'] || data.error;
        document.getElementById('prediction-result').textContent = result;
    })
    .catch(error => {
        document.getElementById('prediction-result').textContent = 'Error: ' + error;
    });
});
