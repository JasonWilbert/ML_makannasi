document.addEventListener('DOMContentLoaded', function() {
    const emailForm = document.getElementById('emailForm');
    const loader = document.getElementById('loader');
    const resultCard = document.getElementById('resultCard');
    const resultTitle = document.getElementById('resultTitle');
    const confidenceLabel = document.getElementById('confidenceLabel'); // Elemen label
    const confidenceText = document.getElementById('confidenceText');
    const confidenceBar = document.getElementById('confidenceBar');
    const explanationList = document.getElementById('explanationList');
    const senderInfo = document.getElementById('senderInfo');
    const dateInfo = document.getElementById('dateInfo');
    const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
    
    emailForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const emailContent = document.getElementById('emailContent').value;
        
        if (!emailContent.trim()) {
            alert('Silakan masukkan konten email');
            return;
        }
        
        // Show loader
        loader.style.display = 'block';
        resultCard.style.display = 'none';
        
        // Send API request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                email_content: emailContent
            })
        })
        .then(response => response.json())
        .then(data => {
            // Hide loader
            loader.style.display = 'none';
            
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            
            // Display results
            displayResults(data);
        })
        .catch(error => {
            // Hide loader
            loader.style.display = 'none';
            console.error('Error:', error);
            alert('Terjadi kesalahan saat memproses permintaan Anda');
        });
    });
    
    analyzeAnotherBtn.addEventListener('click', function() {
        emailForm.reset();
        resultCard.style.display = 'none';
        emailForm.scrollIntoView({ behavior: 'smooth' });
    });
    
    function displayResults(data) {
        // --- PERBAIKAN UTAMA: GANTI LOGIKA PENGECEKAN STATUS ---
        const status = data.prediction_status; // "phishing", "safe", atau "suspicious"
        let confidenceValue;
        let confidencePercent;
        let labelTitle = "Tingkat Kepercayaan"; // Default label

        // Reset kelas CSS terlebih dahulu untuk menghilangkan class sebelumnya
        resultCard.className = 'card result-card';
        confidenceBar.className = 'confidence-fill';

        if (status === 'phishing') {
            // --- ZONA MERAH: PHISHING ---
            resultTitle.innerHTML = '<i class="bi bi-exclamation-triangle-fill text-danger me-2"></i>Email Ini Adalah Phishing';
            resultCard.classList.add('phishing-result');
            confidenceBar.classList.add('phishing-confidence');
            
            // Untuk Phishing, tampilkan probabilitas phishing sebagai tingkat kepercayaan
            confidenceValue = data.phishing_probability;
            labelTitle = "Tingkat Kepercayaan"; // Label tetap "Kepercayaan"

        } else if (status === 'suspicious') {
            // --- ZONA ABU-ABU: MENCURIGAKAN ---
            resultTitle.innerHTML = '<i class="bi bi-exclamation-triangle-fill text-warning me-2"></i>Email Ini Mencurigakan';
            resultCard.classList.add('suspicious-result');
            confidenceBar.classList.add('suspicious-confidence');
            
            // Untuk Mencurigakan, tampilkan probabilitas phishing sebagai TINGKAT KECURIGAAN
            confidenceValue = data.phishing_probability;
            labelTitle = "Tingkat Kecurigaan"; // <--- UBAH LABEL MENJADI "KECURIGAAN"

        } else { // 'safe'
            // --- ZONA HIJAU: AMAN ---
            resultTitle.innerHTML = '<i class="bi bi-check-circle-fill text-success me-2"></i>Email Ini Aman';
            resultCard.classList.add('safe-result');
            confidenceBar.classList.add('safe-confidence');
            
            // Untuk Aman, tampilkan probabilitas aman sebagai tingkat kepercayaan
            confidenceValue = data.safe_probability;
            labelTitle = "Tingkat Kepercayaan"; // Label tetap "Kepercayaan"
        }
        
        // --- PERBAIKAN KEDUA: TAMPILKAN HASIL DENGAN KONTEKS YANG JELAS ---
        // Pastikan confidenceValue adalah angka sebelum dihitung
        confidencePercent = Math.round(confidenceValue * 100);
        
        // Update label, teks, dan progress bar
        confidenceLabel.textContent = labelTitle + ":"; // Ubah teks label
        confidenceText.textContent = confidencePercent + '%';
        confidenceBar.style.width = confidencePercent + '%';
        
        // Set sender and date info
        senderInfo.textContent = data.extracted_sender || 'Tidak terdeteksi';
        dateInfo.textContent = data.extracted_date || 'Tidak terdeteksi';
        
        // Clear and populate explanation list
        explanationList.innerHTML = '';
        data.explanation.forEach(item => {
            const li = document.createElement('li');
            li.className = 'explanation-item';
            li.textContent = item;
            explanationList.appendChild(li);
        });
        
        // Show result card
        resultCard.style.display = 'block';
        resultCard.scrollIntoView({ behavior: 'smooth' });
    }
});