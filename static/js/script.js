console.log("Object Detection frontend loaded");


async function updateStatistics() {

    try {

        const response = await fetch("/stats");

        const data = await response.json();


        // Update total objects
        document.getElementById("object-count").textContent =
            data.object_count;


        // Update tracked objects
        document.getElementById("tracked-count").textContent =
            data.tracked_count;


        // Get object types container
        const objectTypesList =
            document.getElementById("object-types-list");


        // Clear previous object types
        objectTypesList.innerHTML = "";


        const objectTypes = data.object_types;


        // Check if objects exist
        if (Object.keys(objectTypes).length === 0) {

            objectTypesList.innerHTML =
                "<p>No objects detected</p>";

            return;
        }


        // Display every object type
        for (const objectType in objectTypes) {

            const count = objectTypes[objectType];


            const objectItem =
                document.createElement("div");


            objectItem.className =
                "object-type-item";


            objectItem.innerHTML = `
                <span>${objectType}</span>
                <strong>${count}</strong>
            `;


            objectTypesList.appendChild(
                objectItem
            );
        }


    } catch (error) {

        console.error(
            "Error fetching statistics:",
            error
        );

    }
}


// Update statistics every 500 milliseconds
setInterval(
    updateStatistics,
    500
);


// Get statistics immediately
updateStatistics();