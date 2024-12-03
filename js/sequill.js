document.getElementById('bookForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    // Get form values
    var title = document.getElementById('title').value;
    var author = document.getElementById('author').value;
    var description = document.getElementById('description').value;
    var file = document.getElementById('file').value.split('\\').pop(); // Extract the file name

    // Get selected genres (for multiple selection)
    var selectedGenres = Array.from(document.getElementById('genres').selectedOptions)
        .map(function(option) {
            return option.text;
        })
        .join(', ');
    // Get tags
    var tags = document.getElementById('tags').value;

    // Display the entered values
    document.getElementById('bookTitle').textContent = title;
    document.getElementById('bookAuthor').textContent = author;
    document.getElementById('bookDesc').textContent = description;
    document.getElementById('bookGenres').textContent = selectedGenres;
    document.getElementById('bookTags').textContent = tags;
    document.getElementById('bookFile').textContent = file;

    // Show the book display section
    document.getElementById('bookDisplay').style.display = 'block';
});
