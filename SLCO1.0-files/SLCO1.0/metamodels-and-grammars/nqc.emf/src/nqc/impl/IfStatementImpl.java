/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.Expression;
import nqc.IfStatement;
import nqc.NqcPackage;
import nqc.Statement;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>If Statement</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.IfStatementImpl#getCondition <em>Condition</em>}</li>
 *   <li>{@link nqc.impl.IfStatementImpl#getConsequence <em>Consequence</em>}</li>
 *   <li>{@link nqc.impl.IfStatementImpl#getAlternative <em>Alternative</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class IfStatementImpl extends ControlStructureImpl implements IfStatement {
	/**
	 * The cached value of the '{@link #getCondition() <em>Condition</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCondition()
	 * @generated
	 * @ordered
	 */
	protected Expression condition;

	/**
	 * The cached value of the '{@link #getConsequence() <em>Consequence</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getConsequence()
	 * @generated
	 * @ordered
	 */
	protected Statement consequence;

	/**
	 * The cached value of the '{@link #getAlternative() <em>Alternative</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getAlternative()
	 * @generated
	 * @ordered
	 */
	protected Statement alternative;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected IfStatementImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getIfStatement();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Expression getCondition() {
		return condition;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetCondition(Expression newCondition, NotificationChain msgs) {
		Expression oldCondition = condition;
		condition = newCondition;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.IF_STATEMENT__CONDITION, oldCondition, newCondition);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setCondition(Expression newCondition) {
		if (newCondition != condition) {
			NotificationChain msgs = null;
			if (condition != null)
				msgs = ((InternalEObject)condition).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.IF_STATEMENT__CONDITION, null, msgs);
			if (newCondition != null)
				msgs = ((InternalEObject)newCondition).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.IF_STATEMENT__CONDITION, null, msgs);
			msgs = basicSetCondition(newCondition, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.IF_STATEMENT__CONDITION, newCondition, newCondition));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Statement getConsequence() {
		return consequence;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetConsequence(Statement newConsequence, NotificationChain msgs) {
		Statement oldConsequence = consequence;
		consequence = newConsequence;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.IF_STATEMENT__CONSEQUENCE, oldConsequence, newConsequence);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setConsequence(Statement newConsequence) {
		if (newConsequence != consequence) {
			NotificationChain msgs = null;
			if (consequence != null)
				msgs = ((InternalEObject)consequence).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.IF_STATEMENT__CONSEQUENCE, null, msgs);
			if (newConsequence != null)
				msgs = ((InternalEObject)newConsequence).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.IF_STATEMENT__CONSEQUENCE, null, msgs);
			msgs = basicSetConsequence(newConsequence, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.IF_STATEMENT__CONSEQUENCE, newConsequence, newConsequence));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Statement getAlternative() {
		return alternative;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetAlternative(Statement newAlternative, NotificationChain msgs) {
		Statement oldAlternative = alternative;
		alternative = newAlternative;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.IF_STATEMENT__ALTERNATIVE, oldAlternative, newAlternative);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setAlternative(Statement newAlternative) {
		if (newAlternative != alternative) {
			NotificationChain msgs = null;
			if (alternative != null)
				msgs = ((InternalEObject)alternative).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.IF_STATEMENT__ALTERNATIVE, null, msgs);
			if (newAlternative != null)
				msgs = ((InternalEObject)newAlternative).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.IF_STATEMENT__ALTERNATIVE, null, msgs);
			msgs = basicSetAlternative(newAlternative, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.IF_STATEMENT__ALTERNATIVE, newAlternative, newAlternative));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case NqcPackage.IF_STATEMENT__CONDITION:
				return basicSetCondition(null, msgs);
			case NqcPackage.IF_STATEMENT__CONSEQUENCE:
				return basicSetConsequence(null, msgs);
			case NqcPackage.IF_STATEMENT__ALTERNATIVE:
				return basicSetAlternative(null, msgs);
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
			case NqcPackage.IF_STATEMENT__CONDITION:
				return getCondition();
			case NqcPackage.IF_STATEMENT__CONSEQUENCE:
				return getConsequence();
			case NqcPackage.IF_STATEMENT__ALTERNATIVE:
				return getAlternative();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case NqcPackage.IF_STATEMENT__CONDITION:
				setCondition((Expression)newValue);
				return;
			case NqcPackage.IF_STATEMENT__CONSEQUENCE:
				setConsequence((Statement)newValue);
				return;
			case NqcPackage.IF_STATEMENT__ALTERNATIVE:
				setAlternative((Statement)newValue);
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
			case NqcPackage.IF_STATEMENT__CONDITION:
				setCondition((Expression)null);
				return;
			case NqcPackage.IF_STATEMENT__CONSEQUENCE:
				setConsequence((Statement)null);
				return;
			case NqcPackage.IF_STATEMENT__ALTERNATIVE:
				setAlternative((Statement)null);
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
			case NqcPackage.IF_STATEMENT__CONDITION:
				return condition != null;
			case NqcPackage.IF_STATEMENT__CONSEQUENCE:
				return consequence != null;
			case NqcPackage.IF_STATEMENT__ALTERNATIVE:
				return alternative != null;
		}
		return super.eIsSet(featureID);
	}

} //IfStatementImpl
