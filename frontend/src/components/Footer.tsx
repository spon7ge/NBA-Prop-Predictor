export function Footer() {
  return (
    <footer className="site-footer" role="contentinfo">
      <div className="site-footer-inner">
        <div className="site-footer-main">
          <p className="site-footer-disclaimer">
            HoopVista provides research tools, projections, and statistics for informational and
            entertainment purposes only. Nothing on this site is gambling, financial, or legal
            advice. Sports betting involves risk of loss. You are solely responsible for your
            decisions and for complying with applicable laws in your jurisdiction. Past results do
            not guarantee future outcomes.
          </p>
          <p className="site-footer-copy">© Copyright 2026 HoopVista. All Rights Reserved.</p>
        </div>
        <nav className="site-footer-about" aria-labelledby="footer-about-heading">
          <p id="footer-about-heading" className="site-footer-about-title">
            About
          </p>
          <ul className="site-footer-links">
            <li>
              <a href="blog.html">Blog</a>
            </li>
            <li>
              <a href="contact.html">Contact</a>
            </li>
          </ul>
        </nav>
      </div>
    </footer>
  );
}
