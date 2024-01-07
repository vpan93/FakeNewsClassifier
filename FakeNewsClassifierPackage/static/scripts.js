// static/scripts.js
$(document).ready(function() {
    // Function to fetch logs and update the log area
    function fetchLogs() {
        $.get('/get-logs', function(data) {
            $('#log').text(data); // Update the log area with the fetched log data
        }).fail(function() {
            $('#log').text('Failed to fetch logs.'); // Error handling
        });
    }

    // Fetch logs immediately when the document is ready, and then every 5 seconds
    fetchLogs();
    setInterval(fetchLogs, 30000);

    $('#upload-form').on('submit', function(e) {
        e.preventDefault(); // Prevent the default form submission

        // Display a loading message or spinner
        $('#log').html('<div>Loading...</div>');

        var formData = new FormData(this);

        // Make the AJAX request
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false, // tell jQuery not to process the data
            contentType: false, // tell jQuery not to set contentType
            success: function(data) {
                // On success, update the log area with the server's response
                $('#log').html('<div>Upload Successful</div>');
                // If you want to display the JSON result
                $('#results').text(JSON.stringify(data, null, 4));
                fetchLogs(); // Fetch and display the latest logs
            },
            error: function(xhr, status, error) {
                // On error, update the log area with the error message
                $('#log').html('<div>Error during file upload: ' + error + '</div>');
                fetchLogs(); // Fetch and display the latest logs even if there's an error
            }
        });
    });
});
