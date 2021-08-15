/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.GoToStatement;
import nqc.Label;
import nqc.NqcPackage;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Go To Statement</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.GoToStatementImpl#getJumpLabel <em>Jump Label</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class GoToStatementImpl extends ControlStructureImpl implements GoToStatement {
	/**
	 * The cached value of the '{@link #getJumpLabel() <em>Jump Label</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getJumpLabel()
	 * @generated
	 * @ordered
	 */
	protected Label jumpLabel;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected GoToStatementImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getGoToStatement();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Label getJumpLabel() {
		if (jumpLabel != null && jumpLabel.eIsProxy()) {
			InternalEObject oldJumpLabel = (InternalEObject)jumpLabel;
			jumpLabel = (Label)eResolveProxy(oldJumpLabel);
			if (jumpLabel != oldJumpLabel) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, NqcPackage.GO_TO_STATEMENT__JUMP_LABEL, oldJumpLabel, jumpLabel));
			}
		}
		return jumpLabel;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Label basicGetJumpLabel() {
		return jumpLabel;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setJumpLabel(Label newJumpLabel) {
		Label oldJumpLabel = jumpLabel;
		jumpLabel = newJumpLabel;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.GO_TO_STATEMENT__JUMP_LABEL, oldJumpLabel, jumpLabel));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.GO_TO_STATEMENT__JUMP_LABEL:
				if (resolve) return getJumpLabel();
				return basicGetJumpLabel();
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
			case NqcPackage.GO_TO_STATEMENT__JUMP_LABEL:
				setJumpLabel((Label)newValue);
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
			case NqcPackage.GO_TO_STATEMENT__JUMP_LABEL:
				setJumpLabel((Label)null);
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
			case NqcPackage.GO_TO_STATEMENT__JUMP_LABEL:
				return jumpLabel != null;
		}
		return super.eIsSet(featureID);
	}

} //GoToStatementImpl
