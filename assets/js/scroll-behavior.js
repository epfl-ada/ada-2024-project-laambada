document.addEventListener("DOMContentLoaded", function () {
    // Select the top sliding window
    const topWindow = document.getElementById("top-window");
  
    // Add a scroll event listener to the window
    window.addEventListener("scroll", () => {
      // Check how far the user has scrolled
      const scrollPosition = window.scrollY;
  
      // Add or remove the "scrolled" class
      if (scrollPosition > 50) {
        topWindow.classList.add("scrolled");
      } else {
        topWindow.classList.remove("scrolled");
      }
    });
  });
  