document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const modelInput = document.getElementById('selectedModelValue');
    const resultContainer = document.getElementById('resultContainer');
    const resultText = document.getElementById('resultText');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('previewContainer');

    fileInput.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', modelInput.value);

        resultContainer.style.display = 'block';
        resultText.innerHTML = '<div class="spinner-border text-light" role="status"></div><br>Analiza...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Błąd API');

            const data = await response.json();
            
            const confidencePercent = (data.confidence * 100).toFixed(1);
            
            resultText.innerHTML = `
                <h3 class="text-accent fw-bold mb-2">${data.class_name}</h3>
                <div class="small text-white-50">
                    Pewność: <span class="text-white">${confidencePercent}%</span>
                </div>
            `;
        } catch (error) {
            console.error(error);
            resultText.innerHTML = '<span class="text-danger">Wystąpił błąd podczas klasyfikacji.</span>';
        }
    });
});