document.addEventListener('DOMContentLoaded', function() {
    const params = new URLSearchParams(window.location.search);
    const week = params.get('week');
    const lectureContent = document.getElementById('lecture-content');

    const lectures = {
        '1': {
            title: "Week 1: Introduction to Machine Learning",
            description: "Explore the fundamental concepts and types of machine learning, and see real-world applications.",
            image: "path_to_image_1.jpg",
            diagram: "path_to_diagram_1.svg"
        },
        '2': {
            title: "Week 2: Supervised Learning",
            description: "Learn about classification and regression models, with examples in data analysis and prediction.",
            image: "path_to_image_2.jpg",
            diagram: "path_to_diagram_2.svg"
        },
        // Additional weeks can be configured similarly
    };

    if (lectures[week]) {
        lectureContent.innerHTML = `
            <h2 class="text-xl font-bold">${lectures[week].title}</h2>
            <p>${lectures[week].description}</p>
            <img src="${lectures[week].image}" alt="Illustration for ${lectures[week].title}" class="my-4 w-full max-w-lg">
            <object type="image/svg+xml" data="${lectures[week].diagram}" class="w-full max-w-lg">Diagram</object>
        `;
    } else {
        lectureContent.innerHTML = '<p class="text-red-500">Lecture details not found. Please check the schedule and try again.</p>';
    }
});
