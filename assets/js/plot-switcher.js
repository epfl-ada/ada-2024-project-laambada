function switchPlot(index) {
    const plots = document.querySelectorAll('.plot');
    plots.forEach((plot, i) => {
      plot.classList.toggle('active', i === index);
    });
  }
  