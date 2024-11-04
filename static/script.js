document.getElementById('submitBtn').addEventListener('click', function () {
	const query = document.getElementById('query').value;

	// Clear previous answer and recommendations
	document.getElementById('answer').textContent = '';
	document.getElementById('recommendations').innerHTML = '';
	document.getElementById('loader').style.display = 'block'; // Show the loader

	fetch('/api/query', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ query: query }),
	})
		.then((response) => response.json())
		.then((data) => {
			document.getElementById('loader').style.display = 'none'; // Hide the loader

			// Display the response
			document.getElementById('answer').innerHTML = data.answer;

			// Populate recommendations as buttons
			const recommendationsElement = document.getElementById('recommendations');
			data.recommendations.slice(0, 5).forEach((rec) => {
				// Limit to 5 recommendations
				const button = document.createElement('button');
				button.textContent = rec;
				button.className = 'recommendation-button'; // Add class for styling
				button.addEventListener('click', () => {
					// Set the clicked recommendation as the new query
					document.getElementById('query').value = rec;
					// Automatically trigger the submit button to send the query
					document.getElementById('submitBtn').click();
				});
				recommendationsElement.appendChild(button);
			});
		})
		.catch((error) => {
			console.error('Error:', error);
			document.getElementById('loader').style.display = 'none'; // Hide the loader
			document.getElementById('answer').textContent = 'An error occurred.';
		});
});
