import { Link, useLocation } from 'react-router-dom';

function Navbar() {
  const location = useLocation();

  return (
    <nav style={styles.nav}>
      <div style={styles.container}>
        <div style={styles.brand}>
          <Link to="/" style={styles.brandLink}>
            Student Retention <span style={styles.highlight}>AI</span>
          </Link>
        </div>
        <div style={styles.links}>
          <Link 
            to="/" 
            style={{...styles.link, ...(location.pathname === '/' ? styles.activeLink : {})}}
          >
            Prediction
          </Link>
          <Link 
            to="/how-it-works" 
            style={{...styles.link, ...(location.pathname === '/how-it-works' ? styles.activeLink : {})}}
          >
            How it Works
          </Link>
        </div>
      </div>
    </nav>
  );
}

const styles = {
  nav: {
    width: '100%',
    backgroundColor: 'rgba(15, 23, 42, 0.8)',
    backdropFilter: 'blur(12px)',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
    position: 'sticky',
    top: 0,
    zIndex: 100,
  },
  container: {
    maxWidth: '1400px',
    margin: '0 auto',
    padding: '1rem 1.5rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  brand: {
    fontSize: '1.25rem',
    fontWeight: '700',
  },
  brandLink: {
    textDecoration: 'none',
    color: '#f8fafc',
  },
  highlight: {
    color: '#a78bfa',
  },
  links: {
    display: 'flex',
    gap: '2rem',
  },
  link: {
    textDecoration: 'none',
    color: '#94a3b8',
    fontWeight: '500',
    transition: 'color 0.2s',
  },
  activeLink: {
    color: '#a78bfa',
  }
};

export default Navbar;
