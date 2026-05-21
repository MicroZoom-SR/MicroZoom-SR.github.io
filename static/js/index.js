window.HELP_IMPROVE_VIDEOJS = false;

// Scroll to top of page
function topFunction() {
  document.body.scrollTop = 0;
  document.documentElement.scrollTop = 0;
}

$(document).ready(function () {

  // Show/hide back-to-top button on scroll
  let mybutton = document.getElementById("myBtn");
  window.onscroll = function () {
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
      mybutton.style.display = "block";
    } else {
      mybutton.style.display = "none";
    }
  };

  // Navbar burger toggle for mobile
  $(".navbar-burger").click(function () {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  // ── Thumbnail gallery ────────────────────────────────────────────
  // Each .thumb-item carries a data-viewer-src attribute pointing to
  // the per-example URL on the S3 viewer bucket.
  const iframe = document.getElementById("viewer-iframe");
  const thumbs = document.querySelectorAll(".thumb-item");

  // Activate the first thumbnail by default
  if (thumbs.length > 0) {
    thumbs[0].classList.add("is-active");
    iframe.src = thumbs[0].dataset.viewerSrc;
  }

  thumbs.forEach(function (thumb) {
    thumb.addEventListener("click", function () {
      thumbs.forEach(function (t) { t.classList.remove("is-active"); });
      thumb.classList.add("is-active");
      iframe.src = thumb.dataset.viewerSrc;
    });
  });

});
