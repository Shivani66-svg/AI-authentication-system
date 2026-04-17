/* ──────────────────────────────────────────────────────────────
   SECURITIX Showcase — JavaScript
   ────────────────────────────────────────────────────────────── */

// Scroll reveal animation
document.addEventListener('DOMContentLoaded', () => {
    // Mark elements for reveal
    const revealSelectors = [
        '.feature-card', '.arch-card', '.tech-card',
        '.sec-item', '.flow-section', '.qs-step',
        '.code-block', '.qs-requirements', '.section-header'
    ];

    revealSelectors.forEach(sel => {
        document.querySelectorAll(sel).forEach(el => {
            el.classList.add('reveal');
        });
    });

    // Intersection Observer for scroll animations
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                // Stagger the animations
                const delay = Array.from(entry.target.parentElement?.children || [])
                    .indexOf(entry.target) * 80;
                setTimeout(() => {
                    entry.target.classList.add('visible');
                }, delay);
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    document.querySelectorAll('.reveal').forEach(el => {
        observer.observe(el);
    });

    // Navbar scroll effect
    const navbar = document.getElementById('navbar');
    let lastScroll = 0;

    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        if (currentScroll > 100) {
            navbar.style.padding = '10px 0';
            navbar.style.background = 'rgba(10, 11, 16, 0.95)';
        } else {
            navbar.style.padding = '16px 0';
            navbar.style.background = 'rgba(10, 11, 16, 0.8)';
        }

        lastScroll = currentScroll;
    });

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Animate stats on scroll
    const stats = document.querySelectorAll('.stat-number');
    const statsObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                const text = el.textContent;
                const num = parseInt(text);

                if (!isNaN(num) && num <= 100) {
                    let current = 0;
                    const step = Math.max(1, Math.floor(num / 30));
                    const interval = setInterval(() => {
                        current += step;
                        if (current >= num) {
                            current = num;
                            clearInterval(interval);
                        }
                        el.textContent = current;
                    }, 30);
                }

                statsObserver.unobserve(el);
            }
        });
    }, { threshold: 0.5 });

    stats.forEach(stat => statsObserver.observe(stat));
});
