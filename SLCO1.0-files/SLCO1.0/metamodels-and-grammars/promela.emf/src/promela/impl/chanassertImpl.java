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
import promela.chanassert;
import promela.channelassertion;
import promela.varref;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>chanassert</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.chanassertImpl#getChannel_assertion <em>Channel assertion</em>}</li>
 *   <li>{@link promela.impl.chanassertImpl#getVarrefs <em>Varrefs</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class chanassertImpl extends stepImpl implements chanassert {
	/**
	 * The default value of the '{@link #getChannel_assertion() <em>Channel assertion</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getChannel_assertion()
	 * @generated
	 * @ordered
	 */
	protected static final channelassertion CHANNEL_ASSERTION_EDEFAULT = channelassertion.XR;

	/**
	 * The cached value of the '{@link #getChannel_assertion() <em>Channel assertion</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getChannel_assertion()
	 * @generated
	 * @ordered
	 */
	protected channelassertion channel_assertion = CHANNEL_ASSERTION_EDEFAULT;

	/**
	 * The cached value of the '{@link #getVarrefs() <em>Varrefs</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getVarrefs()
	 * @generated
	 * @ordered
	 */
	protected EList<varref> varrefs;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected chanassertImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.CHANASSERT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public channelassertion getChannel_assertion() {
		return channel_assertion;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setChannel_assertion(channelassertion newChannel_assertion) {
		channelassertion oldChannel_assertion = channel_assertion;
		channel_assertion = newChannel_assertion == null ? CHANNEL_ASSERTION_EDEFAULT : newChannel_assertion;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.CHANASSERT__CHANNEL_ASSERTION, oldChannel_assertion, channel_assertion));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<varref> getVarrefs() {
		if (varrefs == null) {
			varrefs = new EObjectContainmentEList<varref>(varref.class, this, PromelaPackage.CHANASSERT__VARREFS);
		}
		return varrefs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.CHANASSERT__VARREFS:
				return ((InternalEList<?>)getVarrefs()).basicRemove(otherEnd, msgs);
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
			case PromelaPackage.CHANASSERT__CHANNEL_ASSERTION:
				return getChannel_assertion();
			case PromelaPackage.CHANASSERT__VARREFS:
				return getVarrefs();
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
			case PromelaPackage.CHANASSERT__CHANNEL_ASSERTION:
				setChannel_assertion((channelassertion)newValue);
				return;
			case PromelaPackage.CHANASSERT__VARREFS:
				getVarrefs().clear();
				getVarrefs().addAll((Collection<? extends varref>)newValue);
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
			case PromelaPackage.CHANASSERT__CHANNEL_ASSERTION:
				setChannel_assertion(CHANNEL_ASSERTION_EDEFAULT);
				return;
			case PromelaPackage.CHANASSERT__VARREFS:
				getVarrefs().clear();
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
			case PromelaPackage.CHANASSERT__CHANNEL_ASSERTION:
				return channel_assertion != CHANNEL_ASSERTION_EDEFAULT;
			case PromelaPackage.CHANASSERT__VARREFS:
				return varrefs != null && !varrefs.isEmpty();
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
		result.append(" (channel_assertion: ");
		result.append(channel_assertion);
		result.append(')');
		return result.toString();
	}

} //chanassertImpl
