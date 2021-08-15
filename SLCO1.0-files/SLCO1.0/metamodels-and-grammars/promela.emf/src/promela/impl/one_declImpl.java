/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import promela.PromelaPackage;
import promela.ivar;
import promela.one_decl;
import promela.typename;
import promela.visibleenum;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>one decl</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.one_declImpl#getVisible <em>Visible</em>}</li>
 *   <li>{@link promela.impl.one_declImpl#getTypename <em>Typename</em>}</li>
 *   <li>{@link promela.impl.one_declImpl#getIvars <em>Ivars</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class one_declImpl extends stepImpl implements one_decl {
	/**
	 * The default value of the '{@link #getVisible() <em>Visible</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getVisible()
	 * @generated
	 * @ordered
	 */
	protected static final visibleenum VISIBLE_EDEFAULT = visibleenum.HIDDEN;

	/**
	 * The cached value of the '{@link #getVisible() <em>Visible</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getVisible()
	 * @generated
	 * @ordered
	 */
	protected visibleenum visible = VISIBLE_EDEFAULT;

	/**
	 * The cached value of the '{@link #getTypename() <em>Typename</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTypename()
	 * @generated
	 * @ordered
	 */
	protected typename typename;

	/**
	 * The cached value of the '{@link #getIvars() <em>Ivars</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getIvars()
	 * @generated
	 * @ordered
	 */
	protected EList<ivar> ivars;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected one_declImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.ONE_DECL;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public visibleenum getVisible() {
		return visible;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setVisible(visibleenum newVisible) {
		visibleenum oldVisible = visible;
		visible = newVisible == null ? VISIBLE_EDEFAULT : newVisible;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.ONE_DECL__VISIBLE, oldVisible, visible));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public typename getTypename() {
		if (typename != null && typename.eIsProxy()) {
			InternalEObject oldTypename = (InternalEObject)typename;
			typename = (typename)eResolveProxy(oldTypename);
			if (typename != oldTypename) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PromelaPackage.ONE_DECL__TYPENAME, oldTypename, typename));
			}
		}
		return typename;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public typename basicGetTypename() {
		return typename;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setTypename(typename newTypename) {
		typename oldTypename = typename;
		typename = newTypename;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.ONE_DECL__TYPENAME, oldTypename, typename));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<ivar> getIvars() {
		if (ivars == null) {
			ivars = new EObjectContainmentEList<ivar>(ivar.class, this, PromelaPackage.ONE_DECL__IVARS);
		}
		return ivars;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.ONE_DECL__IVARS:
				return ((InternalEList<?>)getIvars()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PromelaPackage.ONE_DECL__VISIBLE:
				return getVisible();
			case PromelaPackage.ONE_DECL__TYPENAME:
				if (resolve) return getTypename();
				return basicGetTypename();
			case PromelaPackage.ONE_DECL__IVARS:
				return getIvars();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case PromelaPackage.ONE_DECL__VISIBLE:
				setVisible((visibleenum)newValue);
				return;
			case PromelaPackage.ONE_DECL__TYPENAME:
				setTypename((typename)newValue);
				return;
			case PromelaPackage.ONE_DECL__IVARS:
				getIvars().clear();
				getIvars().addAll((Collection<? extends ivar>)newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case PromelaPackage.ONE_DECL__VISIBLE:
				setVisible(VISIBLE_EDEFAULT);
				return;
			case PromelaPackage.ONE_DECL__TYPENAME:
				setTypename((typename)null);
				return;
			case PromelaPackage.ONE_DECL__IVARS:
				getIvars().clear();
				return;
		}
		super.eUnset(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case PromelaPackage.ONE_DECL__VISIBLE:
				return visible != VISIBLE_EDEFAULT;
			case PromelaPackage.ONE_DECL__TYPENAME:
				return typename != null;
			case PromelaPackage.ONE_DECL__IVARS:
				return ivars != null && !ivars.isEmpty();
		}
		return super.eIsSet(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String toString() {
		if (eIsProxy()) return super.toString();

		StringBuffer result = new StringBuffer(super.toString());
		result.append(" (visible: ");
		result.append(visible);
		result.append(')');
		return result.toString();
	}

} //one_declImpl
