import { getApp, getApps, initializeApp } from 'firebase/app';
import {
  browserLocalPersistence,
  createUserWithEmailAndPassword,
  getAuth,
  onAuthStateChanged,
  setPersistence,
  signInWithEmailAndPassword,
  signOut,
  updateProfile,
} from 'firebase/auth';
import { firebaseConfig } from './firebase-config.js';

const app = getApps().length ? getApp() : initializeApp(firebaseConfig);
const auth = getAuth(app);
window.urjaGetIdToken = async () => auth.currentUser ? auth.currentUser.getIdToken() : null;

function messageFor(error) {
  const messages = {
    'auth/email-already-in-use': 'An account already exists for this email. Sign in instead.',
    'auth/invalid-email': 'Enter a valid email address.',
    'auth/weak-password': 'Choose a stronger password (at least 6 characters).',
    'auth/invalid-credential': 'Email or password is incorrect.',
    'auth/user-disabled': 'This account has been disabled. Contact your administrator.',
    'auth/too-many-requests': 'Too many attempts. Wait a little and try again.',
    'auth/network-request-failed': 'Could not reach Firebase. Check your internet connection.',
    'auth/operation-not-allowed': 'Email/password sign-in is not enabled in Firebase Authentication.',
    'auth/unauthorized-domain': 'This website domain is not authorized in Firebase Authentication settings.',
  };
  return messages[error?.code] || error?.message || 'Firebase Authentication could not complete the request.';
}

window.submitAuth = async function submitFirebaseAuth() {
  const mode = window.urjaAuthMode();
  const email = document.getElementById('auth-email')?.value.trim();
  const password = document.getElementById('auth-password')?.value;
  if (!email || !password) {
    window.urjaToast('Enter your email and password.', 'warn');
    return;
  }
  if (mode === 'signup' && password.length < 8) {
    window.urjaToast('Use a password with at least 8 characters.', 'warn');
    return;
  }

  const button = document.querySelector('.auth-card .button.full');
  if (button) {
    button.disabled = true;
    button.textContent = mode === 'signup' ? 'Creating account…' : 'Signing in…';
  }

  try {
    if (mode === 'signup') {
      const name = document.getElementById('auth-name')?.value.trim() || 'Factory Admin';
      const factory = document.getElementById('auth-factory')?.value.trim();
      const registerDevice = !!document.getElementById('auth-register')?.checked;
      const credential = await createUserWithEmailAndPassword(auth, email, password);
      await updateProfile(credential.user, { displayName: name });
      window.urjaSetSession(credential.user);
      if (factory) window.urjaSetFactory(factory);
      if (registerDevice) window.urjaRegisterThisDevice();
    } else {
      const credential = await signInWithEmailAndPassword(auth, email, password);
      window.urjaSetSession(credential.user);
    }
    window.closeAuth();
  } catch (error) {
    window.urjaToast(messageFor(error), 'error');
    if (button) {
      button.disabled = false;
      button.textContent = mode === 'signup' ? 'Create account' : 'Sign in';
    }
  }
};

window.logout = async function logoutFromFirebase() {
  try {
    await signOut(auth);
  } catch (error) {
    window.urjaToast(messageFor(error), 'error');
  }
};

window.firebaseAuthReady = setPersistence(auth, browserLocalPersistence)
  .then(() => new Promise((resolve) => {
    let unsubscribe = () => {};
    unsubscribe = onAuthStateChanged(auth, (user) => {
      window.urjaSetSession(user);
      unsubscribe();
      resolve(user);
    }, (error) => {
      window.urjaSetSession(null);
      window.urjaToast(messageFor(error), 'error');
      unsubscribe();
      resolve(null);
    });
  }))
  .catch((error) => {
    window.urjaSetSession(null);
    window.urjaToast(messageFor(error), 'error');
    return null;
  });
