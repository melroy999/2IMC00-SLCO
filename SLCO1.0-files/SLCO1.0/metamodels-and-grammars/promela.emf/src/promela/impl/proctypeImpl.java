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
import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import promela.PromelaPackage;
import promela.expr;
import promela.one_decl;
import promela.proctype;
import promela.step;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>proctype</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.proctypeImpl#getActive <em>Active</em>}</li>
 *   <li>{@link promela.impl.proctypeImpl#getName <em>Name</em>}</li>
 *   <li>{@link promela.impl.proctypeImpl#getOne_decls <em>One decls</em>}</li>
 *   <li>{@link promela.impl.proctypeImpl#getPriority <em>Priority</em>}</li>
 *   <li>{@link promela.impl.proctypeImpl#getEnabler <em>Enabler</em>}</li>
 *   <li>{@link promela.impl.proctypeImpl#getSequence <em>Sequence</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class proctypeImpl extends EObjectImpl implements proctype {
	/**
	 * The default value of the '{@link #getActive() <em>Active</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getActive()
	 * @generated
	 * @ordered
	 */
	protected static final int ACTIVE_EDEFAULT = 0;

	/**
	 * The cached value of the '{@link #getActive() <em>Active</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getActive()
	 * @generated
	 * @ordered
	 */
	protected int active = ACTIVE_EDEFAULT;

	/**
	 * The default value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected static final String NAME_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected String name = NAME_EDEFAULT;

	/**
	 * The cached value of the '{@link #getOne_decls() <em>One decls</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOne_decls()
	 * @generated
	 * @ordered
	 */
	protected EList<one_decl> one_decls;

	/**
	 * The default value of the '{@link #getPriority() <em>Priority</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPriority()
	 * @generated
	 * @ordered
	 */
	protected static final int PRIORITY_EDEFAULT = 0;

	/**
	 * The cached value of the '{@link #getPriority() <em>Priority</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPriority()
	 * @generated
	 * @ordered
	 */
	protected int priority = PRIORITY_EDEFAULT;

	/**
	 * The cached value of the '{@link #getEnabler() <em>Enabler</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getEnabler()
	 * @generated
	 * @ordered
	 */
	protected expr enabler;

	/**
	 * The cached value of the '{@link #getSequence() <em>Sequence</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSequence()
	 * @generated
	 * @ordered
	 */
	protected EList<step> sequence;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected proctypeImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.PROCTYPE;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public int getActive() {
		return active;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setActive(int newActive) {
		int oldActive = active;
		active = newActive;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.PROCTYPE__ACTIVE, oldActive, active));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getName() {
		return name;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setName(String newName) {
		String oldName = name;
		name = newName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.PROCTYPE__NAME, oldName, name));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<one_decl> getOne_decls() {
		if (one_decls == null) {
			one_decls = new EObjectContainmentEList<one_decl>(one_decl.class, this, PromelaPackage.PROCTYPE__ONE_DECLS);
		}
		return one_decls;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public int getPriority() {
		return priority;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setPriority(int newPriority) {
		int oldPriority = priority;
		priority = newPriority;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.PROCTYPE__PRIORITY, oldPriority, priority));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public expr getEnabler() {
		return enabler;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetEnabler(expr newEnabler, NotificationChain msgs) {
		expr oldEnabler = enabler;
		enabler = newEnabler;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PromelaPackage.PROCTYPE__ENABLER, oldEnabler, newEnabler);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setEnabler(expr newEnabler) {
		if (newEnabler != enabler) {
			NotificationChain msgs = null;
			if (enabler != null)
				msgs = ((InternalEObject)enabler).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.PROCTYPE__ENABLER, null, msgs);
			if (newEnabler != null)
				msgs = ((InternalEObject)newEnabler).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.PROCTYPE__ENABLER, null, msgs);
			msgs = basicSetEnabler(newEnabler, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.PROCTYPE__ENABLER, newEnabler, newEnabler));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<step> getSequence() {
		if (sequence == null) {
			sequence = new EObjectContainmentEList<step>(step.class, this, PromelaPackage.PROCTYPE__SEQUENCE);
		}
		return sequence;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.PROCTYPE__ONE_DECLS:
				return ((InternalEList<?>)getOne_decls()).basicRemove(otherEnd, msgs);
			case PromelaPackage.PROCTYPE__ENABLER:
				return basicSetEnabler(null, msgs);
			case PromelaPackage.PROCTYPE__SEQUENCE:
				return ((InternalEList<?>)getSequence()).basicRemove(otherEnd, msgs);
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
			case PromelaPackage.PROCTYPE__ACTIVE:
				return getActive();
			case PromelaPackage.PROCTYPE__NAME:
				return getName();
			case PromelaPackage.PROCTYPE__ONE_DECLS:
				return getOne_decls();
			case PromelaPackage.PROCTYPE__PRIORITY:
				return getPriority();
			case PromelaPackage.PROCTYPE__ENABLER:
				return getEnabler();
			case PromelaPackage.PROCTYPE__SEQUENCE:
				return getSequence();
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
			case PromelaPackage.PROCTYPE__ACTIVE:
				setActive((Integer)newValue);
				return;
			case PromelaPackage.PROCTYPE__NAME:
				setName((String)newValue);
				return;
			case PromelaPackage.PROCTYPE__ONE_DECLS:
				getOne_decls().clear();
				getOne_decls().addAll((Collection<? extends one_decl>)newValue);
				return;
			case PromelaPackage.PROCTYPE__PRIORITY:
				setPriority((Integer)newValue);
				return;
			case PromelaPackage.PROCTYPE__ENABLER:
				setEnabler((expr)newValue);
				return;
			case PromelaPackage.PROCTYPE__SEQUENCE:
				getSequence().clear();
				getSequence().addAll((Collection<? extends step>)newValue);
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
			case PromelaPackage.PROCTYPE__ACTIVE:
				setActive(ACTIVE_EDEFAULT);
				return;
			case PromelaPackage.PROCTYPE__NAME:
				setName(NAME_EDEFAULT);
				return;
			case PromelaPackage.PROCTYPE__ONE_DECLS:
				getOne_decls().clear();
				return;
			case PromelaPackage.PROCTYPE__PRIORITY:
				setPriority(PRIORITY_EDEFAULT);
				return;
			case PromelaPackage.PROCTYPE__ENABLER:
				setEnabler((expr)null);
				return;
			case PromelaPackage.PROCTYPE__SEQUENCE:
				getSequence().clear();
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
			case PromelaPackage.PROCTYPE__ACTIVE:
				return active != ACTIVE_EDEFAULT;
			case PromelaPackage.PROCTYPE__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case PromelaPackage.PROCTYPE__ONE_DECLS:
				return one_decls != null && !one_decls.isEmpty();
			case PromelaPackage.PROCTYPE__PRIORITY:
				return priority != PRIORITY_EDEFAULT;
			case PromelaPackage.PROCTYPE__ENABLER:
				return enabler != null;
			case PromelaPackage.PROCTYPE__SEQUENCE:
				return sequence != null && !sequence.isEmpty();
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
		result.append(" (active: ");
		result.append(active);
		result.append(", name: ");
		result.append(name);
		result.append(", priority: ");
		result.append(priority);
		result.append(')');
		return result.toString();
	}

} //proctypeImpl
