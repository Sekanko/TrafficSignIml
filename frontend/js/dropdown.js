document.addEventListener('DOMContentLoaded', function() {
    
    // Pobieramy elementy
    const wrapper = document.querySelector('.custom-select-wrapper');
    const trigger = document.querySelector('.custom-select-trigger');
    const triggerText = trigger.querySelector('span');
    const options = document.querySelectorAll('.custom-option');
    const hiddenInput = document.getElementById('selectedModelValue');

    // 1. Otwieranie / Zamykanie po kliknięciu w trigger
    trigger.addEventListener('click', function() {
        wrapper.classList.toggle('open');
    });

    // 2. Obsługa wyboru opcji
    options.forEach(option => {
        option.addEventListener('click', function() {
            // Zaznacz wizualnie wybraną opcję
            options.forEach(opt => opt.classList.remove('selected'));
            this.classList.add('selected');

            // Zmień tekst w triggerze na tekst opcji
            triggerText.textContent = this.textContent;

            // Zapisz wartość (data-value) do ukrytego inputa
            // To tę wartość pobierzesz potem w Pythonie/JS
            const value = this.getAttribute('data-value');
            hiddenInput.value = value;
            console.log("Wybrano model:", value);

            // Zamknij dropdown
            wrapper.classList.remove('open');
        });
    });

    // 3. Zamknij dropdown, jeśli klikniemy gdziekolwiek poza nim
    document.addEventListener('click', function(e) {
        if (!wrapper.contains(e.target)) {
            wrapper.classList.remove('open');
        }
    });

});