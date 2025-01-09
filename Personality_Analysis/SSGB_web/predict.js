document.getElementById('userForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const userInput = document.getElementById('userData').value;

    try {
        const jsonData = JSON.parse(userInput); // Validate JSON input
        const labels = await fetchPredictions(jsonData); // Fetch predictions from the server
        displayResults(labels);
    } catch (err) {
        alert('Invalid JSON format or server error. Please check your input.');
    }
});

async function fetchPredictions(data) {
    try {
        const response = await fetch('http://127.0.0.1:5000/predict', { // 修改为本地 URL
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        return result.labels; // Assuming the backend returns { "labels": [...] }
    } catch (err) {
        console.error('Error fetching predictions:', err);
        alert('Failed to fetch predictions. Please try again later.');
        throw err;
    }
}

function displayResults(labels) {
    document.getElementById('result').innerText = JSON.stringify(labels, null, 2);
    document.getElementById('resultContainer').style.display = 'block';
}

function goBack() {
    window.location.href = 'index.html'; // Redirect to homepage
}
