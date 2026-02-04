document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const modelInput = document.getElementById('selectedModelValue');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('previewContainer');
    const uploadText = document.getElementById('uploadText');
    const spinner = document.getElementById('loadingSpinner');

    fileInput.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        // Pokaż, że coś się dzieje
        uploadText.textContent = "Przetwarzanie...";
        previewContainer.style.display = 'block';
        imagePreview.style.opacity = '0.3'; // Przyciemnij stare zdjęcie (jeśli było)
        spinner.style.display = 'inline-block';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', modelInput.value);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(errText || 'Błąd serwera');
            }

            // Odbieramy obrazek jako Blob
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);

            // Podmieniamy src obrazka
            imagePreview.src = imageUrl;
            imagePreview.style.opacity = '1';
            
        } catch (error) {
            console.error(error);
            alert("Wystąpił błąd podczas detekcji: " + error.message);
            previewContainer.style.display = 'none';
        } finally {
            uploadText.textContent = "Prześlij kolejne zdjęcie";
            spinner.style.display = 'none';
            // Czyścimy input, żeby można było wgrać ten sam plik ponownie
            fileInput.value = '';
        }
    });
});