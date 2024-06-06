function detectAbuse() {
    const comment = document.getElementById('comment').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ comment: comment })
    })
    .then(response => response.json())
    .then(data => {
        const resultElement = document.getElementById('result');
        resultElement.textContent = data.result ? 'Abusive' : 'Not Abusive';
        resultElement.style.color = data.result ? 'red' : 'green';
    })
    .catch(error => console.error('Error:', error));
}

